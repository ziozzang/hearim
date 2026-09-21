package selfupdate

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"
)

// NotifyConfig controls the background update notice (hftools pattern).
type NotifyConfig struct {
	Repo        string
	Current     string
	CheckEvery  time.Duration
	CachePathFn func() string
	// Now is injectable for tests.
	Now func() time.Time
}

// CheckInterval is the default network-check throttle.
const CheckInterval = 24 * time.Hour

// NotifyCache is the persisted result of the last background check.
type NotifyCache struct {
	CheckedAt time.Time `json:"checked_at"`
	Latest    string    `json:"latest_version"`
}

// DefaultCachePath resolves ~/.cache/hearim/update-check.json (or the
// platform equivalent via os.UserCacheDir).
func DefaultCachePath() string {
	dir, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "hearim", "update-check.json")
}

// ExemptCommands never show an update notice: the notice would be noise or
// redundant there.
var ExemptCommands = map[string]bool{
	"": true, "update": true, "self-update": true, "version": true,
	"--version": true, "-version": true, "-v": true, "-V": true,
	"help": true, "-h": true, "--help": true, "completion": true,
}

// NotifyEligible reports whether the update notice machinery should run:
// disabled via HEARIM_NO_UPDATE_CHECK, skipped for exempt commands, and
// silent when stderr is not a terminal (scripts, pipes, service logs).
func NotifyEligible(args []string) bool {
	if os.Getenv("HEARIM_NO_UPDATE_CHECK") != "" {
		return false
	}
	cmd := ""
	if len(args) > 0 {
		cmd = args[0]
	}
	if ExemptCommands[cmd] {
		return false
	}
	return StderrIsTerminal()
}

// StderrIsTerminal reports whether stderr is a character device; it is
// implemented per-OS in stderr_*.go files.
var StderrIsTerminal = func() bool { return stderrIsTerminal() }

// StartNotifyRefresh kicks off a background version check when the cached
// result is stale, then returns immediately. The foreground never waits, so
// offline machines are never slowed: the goroutine soft-fails and writes
// nothing. A successful check refreshes the cache the notice reads.
func StartNotifyRefresh(cfg NotifyConfig) {
	path := cachePathOrDefault(cfg)
	if path == "" {
		return
	}
	c := readNotifyCache(path)
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	if c.Latest != "" && now().Sub(c.CheckedAt) < intervalOrDefault(cfg) {
		printNotice(cfg, c.Latest)
		return
	}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		rel, err := LatestRelease(ctx, &http.Client{}, "", cfg.Repo, os.Getenv("GITHUB_TOKEN"))
		if err != nil {
			return // soft-fail: no cache write, no output
		}
		_ = writeNotifyCache(path, NotifyCache{CheckedAt: now().UTC(), Latest: rel.Version()})
	}()
	// Show a stale-cached notice immediately; the refresh lands next run.
	if c.Latest != "" {
		printNotice(cfg, c.Latest)
	}
	_ = runtime.GOOS
}

func printNotice(cfg NotifyConfig, latest string) {
	if CompareVersions(latest, cfg.Current) > 0 {
		fmt.Fprintf(os.Stderr, "hearim %s is available (current %s); run `hearim update` to upgrade.\n",
			latest, cfg.Current)
	}
}

func cachePathOrDefault(cfg NotifyConfig) string {
	if cfg.CachePathFn != nil {
		return cfg.CachePathFn()
	}
	return DefaultCachePath()
}

func intervalOrDefault(cfg NotifyConfig) time.Duration {
	if cfg.CheckEvery > 0 {
		return cfg.CheckEvery
	}
	return CheckInterval
}

func readNotifyCache(path string) NotifyCache {
	var c NotifyCache
	data, err := os.ReadFile(path)
	if err != nil {
		return c
	}
	_ = json.Unmarshal(data, &c)
	return c
}

func writeNotifyCache(path string, c NotifyCache) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.Marshal(c)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
