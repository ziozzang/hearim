// Package usage implements token/cost accounting of TODO.md §8.5 and the
// budget guard of §9.1. Reported, estimated, and billed quantities are kept
// strictly separate; estimates are never surfaced as billing facts.
package usage

import (
	"sync"
	"time"

	"hearim/internal/hearim/config"
)

// Aggregate is the usage line for one request or window (§8.5).
type Aggregate struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	// ReportedCachedTokens only counts values the server reported.
	ReportedCachedTokens int64 `json:"reported_cached_tokens"`
	// EstimatedCachedTokens is an estimation from prefix-hash and latency —
	// never mixed into billed figures.
	EstimatedCachedTokens int64 `json:"estimated_cached_tokens"`
	UpstreamCalls         int64 `json:"upstream_calls"`
}

// Add accumulates another aggregate.
func (a *Aggregate) Add(o Aggregate) {
	a.InputTokens += o.InputTokens
	a.OutputTokens += o.OutputTokens
	a.ReportedCachedTokens += o.ReportedCachedTokens
	a.EstimatedCachedTokens += o.EstimatedCachedTokens
	a.UpstreamCalls += o.UpstreamCalls
}

// Ledger records internal cost per route (§8.5): local engines carry no
// provider billing, so the ledger keeps the decomposition a virtual input
// rate would need.
type Ledger struct {
	mu sync.Mutex
	// by route: uncached prefill, cached prefill, decode tokens, seconds.
	rows map[string]*LedgerRow
}

type LedgerRow struct {
	UncachedPrefillTokens int64
	CachedPrefillTokens   int64
	DecodeTokens          int64
	AcceleratorSeconds    float64
	USD                   float64
}

func NewLedger() *Ledger { return &Ledger{rows: map[string]*LedgerRow{}} }

// Record adds one observation for a route.
func (l *Ledger) Record(route string, agg Aggregate, accelSeconds float64, usd float64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	row := l.rows[route]
	if row == nil {
		row = &LedgerRow{}
		l.rows[route] = row
	}
	cached := agg.ReportedCachedTokens // billed basis: reported only
	row.UncachedPrefillTokens += agg.InputTokens - min64(cached, agg.InputTokens)
	row.CachedPrefillTokens += cached
	row.DecodeTokens += agg.OutputTokens
	row.AcceleratorSeconds += accelSeconds
	row.USD += usd
}

// Snapshot copies the current rows.
func (l *Ledger) Snapshot() map[string]LedgerRow {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make(map[string]LedgerRow, len(l.rows))
	for k, v := range l.rows {
		out[k] = *v
	}
	return out
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// BudgetState is the guard outcome for a new request (§9.1).
type BudgetState int

const (
	BudgetOK BudgetState = iota
	BudgetWarn
	BudgetThrottle
	BudgetHardStop
)

func (s BudgetState) String() string {
	switch s {
	case BudgetOK:
		return "ok"
	case BudgetWarn:
		return "warn"
	case BudgetThrottle:
		return "throttle"
	default:
		return "hard_stop"
	}
}

// Budget tracks spend against the included monthly allowance.
type Budget struct {
	cfg         config.BudgetConfig
	mu          sync.Mutex
	spentUSD    float64
	windowStart time.Time
}

func NewBudget(cfg config.BudgetConfig) *Budget {
	return &Budget{cfg: cfg, windowStart: time.Now()}
}

// Record adds spend; the window resets on calendar month boundaries.
func (b *Budget) Record(usd float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if time.Since(b.windowStart) > 31*24*time.Hour {
		b.spentUSD = 0
		b.windowStart = time.Now()
	}
	b.spentUSD += usd
}

// State classifies current utilization.
func (b *Budget) State() (BudgetState, float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.cfg.IncludedUSD <= 0 {
		return BudgetOK, 0
	}
	frac := b.spentUSD / b.cfg.IncludedUSD
	switch {
	case frac >= b.cfg.HardStopAt:
		return BudgetHardStop, frac
	case frac >= b.cfg.ThrottleAt:
		return BudgetThrottle, frac
	case frac >= b.cfg.WarnAt:
		return BudgetWarn, frac
	default:
		return BudgetOK, frac
	}
}

// Spent reports current spend.
func (b *Budget) Spent() float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.spentUSD
}
