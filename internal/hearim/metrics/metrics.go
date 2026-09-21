// Package metrics implements a minimal Prometheus text-exposition registry
// (stdlib only). hearim avoids client_golang to keep the dependency surface
// at stdlib + yaml.v3; the subset here — counters, gauges, and a fixed-bucket
// histogram — covers the gateway's operational surface.
package metrics

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

// DefaultBuckets are the histogram boundaries for request durations.
var DefaultBuckets = []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10}

// Registry holds all metric families.
type Registry struct {
	mu         sync.Mutex
	counters   map[string]*family
	gauges     map[string]*family
	histograms map[string]*histogramFamily
}

type family struct {
	help   string
	series map[string]*seriesValue // key: sorted label string
}

type seriesValue struct {
	labels []string // ordered k=v pairs
	value  uint64   // counter/gauge integral part is read under lock
	fvalue float64
}

type histogramFamily struct {
	help    string
	buckets []float64
	series  map[string]*histogramSeries
}

type histogramSeries struct {
	labels []string
	counts []uint64 // per bucket (cumulative at render)
	count  uint64
	sum    float64
}

// New creates an empty registry.
func New() *Registry {
	return &Registry{
		counters:   map[string]*family{},
		gauges:     map[string]*family{},
		histograms: map[string]*histogramFamily{},
	}
}

func labelKey(labels map[string]string) string {
	if len(labels) == 0 {
		return ""
	}
	keys := make([]string, 0, len(labels))
	for k := range labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+escapeLabel(labels[k]))
	}
	return strings.Join(parts, ",")
}

func orderedLabels(labels map[string]string) []string {
	keys := make([]string, 0, len(labels))
	for k := range labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+`="`+escapeLabel(labels[k])+`"`)
	}
	return parts
}

func escapeLabel(v string) string {
	v = strings.ReplaceAll(v, `\`, `\\`)
	v = strings.ReplaceAll(v, `"`, `\"`)
	v = strings.ReplaceAll(v, "\n", `\n`)
	return v
}

// Counter returns (creating if needed) a counter series.
func (r *Registry) Counter(name, help string, labels map[string]string) *CounterV {
	r.mu.Lock()
	defer r.mu.Unlock()
	f := r.counters[name]
	if f == nil {
		f = &family{help: help, series: map[string]*seriesValue{}}
		r.counters[name] = f
	}
	key := labelKey(labels)
	sv := f.series[key]
	if sv == nil {
		sv = &seriesValue{labels: orderedLabels(labels)}
		f.series[key] = sv
	}
	return &CounterV{reg: r, sv: sv}
}

// Gauge returns (creating if needed) a gauge series.
func (r *Registry) Gauge(name, help string, labels map[string]string) *GaugeV {
	r.mu.Lock()
	defer r.mu.Unlock()
	f := r.gauges[name]
	if f == nil {
		f = &family{help: help, series: map[string]*seriesValue{}}
		r.gauges[name] = f
	}
	key := labelKey(labels)
	sv := f.series[key]
	if sv == nil {
		sv = &seriesValue{labels: orderedLabels(labels)}
		f.series[key] = sv
	}
	return &GaugeV{reg: r, sv: sv}
}

// Observe records a duration observation into a histogram family.
func (r *Registry) Observe(name, help string, labels map[string]string, v float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	f := r.histograms[name]
	if f == nil {
		f = &histogramFamily{help: help, buckets: DefaultBuckets, series: map[string]*histogramSeries{}}
		r.histograms[name] = f
	}
	key := labelKey(labels)
	hs := f.series[key]
	if hs == nil {
		hs = &histogramSeries{labels: orderedLabels(labels), counts: make([]uint64, len(f.buckets))}
		f.series[key] = hs
	}
	for i, b := range f.buckets {
		if v <= b {
			hs.counts[i]++
		}
	}
	hs.count++
	hs.sum += v
}

// CounterV is an incrementing counter handle.
type CounterV struct {
	reg *Registry
	sv  *seriesValue
}

// Inc adds one.
func (c *CounterV) Inc() { c.Add(1) }

// Add adds n.
func (c *CounterV) Add(n uint64) {
	c.reg.mu.Lock()
	c.sv.value += n
	c.reg.mu.Unlock()
}

// GaugeV is a settable gauge handle.
type GaugeV struct {
	reg *Registry
	sv  *seriesValue
}

// Set replaces the value.
func (g *GaugeV) Set(v float64) {
	g.reg.mu.Lock()
	g.sv.fvalue = v
	g.reg.mu.Unlock()
}

// Render emits the Prometheus text exposition format.
func (r *Registry) Render() string {
	var b strings.Builder

	r.mu.Lock()
	defer r.mu.Unlock()

	names := make([]string, 0, len(r.counters))
	for n := range r.counters {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		f := r.counters[n]
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s counter\n", n, f.help, n)
		for _, sv := range f.series {
			renderSample(&b, n, sv.labels, float64(sv.value))
		}
	}

	names = names[:0]
	for n := range r.gauges {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		f := r.gauges[n]
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s gauge\n", n, f.help, n)
		for _, sv := range f.series {
			renderSample(&b, n, sv.labels, sv.fvalue)
		}
	}

	names = names[:0]
	for n := range r.histograms {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		f := r.histograms[n]
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s histogram\n", n, f.help, n)
		// Bucket series first, ordered by label key for stability.
		keys := make([]string, 0, len(f.series))
		for k := range f.series {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			hs := f.series[k]
			for i, bound := range f.buckets {
				renderBucket(&b, n, hs.labels, formatValue(bound), hs.counts[i])
			}
			renderBucket(&b, n, hs.labels, "+Inf", hs.count)
			renderSampleSuffix(&b, n+"_sum", filterLe(hs.labels), hs.sum)
			renderSampleSuffix(&b, n+"_count", filterLe(hs.labels), float64(hs.count))
		}
	}
	return b.String()
}

func filterLe(labels []string) []string {
	out := make([]string, 0, len(labels))
	for _, l := range labels {
		if strings.HasPrefix(l, "le=") {
			continue
		}
		out = append(out, l)
	}
	return out
}

func renderSample(b *strings.Builder, name string, labels []string, v float64) {
	renderSampleSuffix(b, name, labels, v)
}

func renderSampleSuffix(b *strings.Builder, name string, labels []string, v float64) {
	if len(labels) == 0 {
		fmt.Fprintf(b, "%s %s\n", name, formatValue(v))
		return
	}
	fmt.Fprintf(b, "%s{%s} %s\n", name, strings.Join(labels, ","), formatValue(v))
}

func renderBucket(b *strings.Builder, name string, labels []string, le string, count uint64) {
	all := append(filterLe(labels), `le="`+le+`"`)
	fmt.Fprintf(b, "%s_bucket{%s} %d\n", name, strings.Join(all, ","), count)
}

func formatValue(v float64) string {
	if v == float64(uint64(v)) && v < 1e15 {
		return fmt.Sprintf("%d", uint64(v))
	}
	return fmt.Sprintf("%g", v)
}
