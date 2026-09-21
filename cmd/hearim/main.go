// Command hearim is the Jev-compatible multi-backend System One gateway
// (TODO.md). Subcommands:
//
//	hearim serve  — run the gateway HTTP server
//	hearim probe  — run the Phase 0 capability suite against providers
//	hearim bench  — run a labeled corpus benchmark (§12.3)
//	hearim version
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"hearim/internal/hearim/bench"
	"hearim/internal/hearim/config"
	"hearim/internal/hearim/httpapi"
	"hearim/internal/hearim/probe"
	"hearim/internal/hearim/provider"
	"hearim/internal/hearim/selfupdate"
)

var version = "0.6.0"

// updateRepo is the GitHub repository self-update pulls release builds from.
const updateRepo = "ziozzang/hearim"

func main() {
	// Background update notice (hftools pattern): terminal-only, throttled
	// to one network check per day, disabled via HEARIM_NO_UPDATE_CHECK.
	selfupdate.StartNotifyRefresh(selfupdate.NotifyConfig{
		Repo:    updateRepo,
		Current: version,
	})

	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "serve":
		err = cmdServe(os.Args[2:])
	case "probe":
		err = cmdProbe(os.Args[2:])
	case "bench":
		err = cmdBench(os.Args[2:])
	case "update":
		err = cmdUpdate(os.Args[2:])
	case "version":
		fmt.Println("hearim", version)
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "hearim:", err)
		os.Exit(1)
	}
}

// cmdUpdate replaces the running binary with the latest (or a specified)
// GitHub release build, verifying it against the release SHA256SUMS first.
func cmdUpdate(args []string) error {
	fs := flag.NewFlagSet("update", flag.ExitOnError)
	check := fs.Bool("check", false, "only report whether a newer version is available")
	force := fs.Bool("force", false, "reinstall even if already on the latest version")
	targetVer := fs.String("version", "", "install a specific release tag (for example v0.2.0) instead of the latest")
	repo := fs.String("repo", updateRepo, "GitHub owner/repo to update from")
	if err := fs.Parse(args); err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	client := &http.Client{}
	token := os.Getenv("GITHUB_TOKEN")

	var rel *selfupdate.Release
	var err error
	if *targetVer != "" {
		rel, err = selfupdate.ReleaseByTag(ctx, client, "", *repo, *targetVer, token)
	} else {
		rel, err = selfupdate.LatestRelease(ctx, client, "", *repo, token)
	}
	if err != nil {
		return fmt.Errorf("look up release: %w", err)
	}
	latest := rel.Version()
	cmp := selfupdate.CompareVersions(latest, version)

	fmt.Printf("current: %s\nlatest:  %s\n", version, latest)
	switch {
	case cmp > 0:
		fmt.Printf("a newer version is available: %s -> %s\n", version, latest)
	case cmp == 0:
		fmt.Println("you are on the latest version")
	default:
		fmt.Printf("your version is newer than the published release (%s)\n", latest)
	}
	if *check {
		return nil
	}
	if cmp <= 0 && *targetVer == "" && !*force {
		return nil
	}

	assetName, err := selfupdate.CurrentAssetName(latest)
	if err != nil {
		return err
	}
	asset, ok := rel.FindAsset(assetName)
	if !ok {
		return fmt.Errorf("release %s has no build for %s/%s (asset %q)", rel.TagName, runtime.GOOS, runtime.GOARCH, assetName)
	}
	sums, err := selfupdate.Checksums(ctx, client, rel, token)
	if err != nil {
		return err
	}
	want := sums[assetName]

	exe, err := selfupdate.ResolveExecutable()
	if err != nil {
		return err
	}
	tmp, err := selfupdate.DownloadVerified(ctx, client, asset, want, exeDir(exe), token)
	if err != nil {
		return err
	}
	defer os.Remove(tmp)
	if err := selfupdate.ReplaceExecutable(exe, tmp); err != nil {
		return err
	}
	fmt.Printf("updated %s -> %s (%s)\n", version, latest, exe)
	return nil
}

func exeDir(p string) string {
	dir := p
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' || p[i] == '\\' {
			dir = p[:i]
			break
		}
	}
	if dir == p {
		dir = "."
	}
	return dir
}

func usage() {
	fmt.Fprint(os.Stderr, `usage: hearim <command> [flags]

commands:
  serve   run the gateway HTTP server
  probe   run capability probes against configured providers (Phase 0)
  bench   run a labeled JSONL corpus benchmark
  update  replace this binary with the latest GitHub release build
  version print version
`)
}

func commonFlags(fs *flag.FlagSet) (configPath *string, logLevel *string, logJSON *bool) {
	configPath = fs.String("config", "hearim.yaml", "path to YAML configuration")
	logLevel = fs.String("log-level", "", "debug|info|warn|error (overrides config)")
	logJSON = fs.Bool("log-json", false, "JSON log output (overrides config)")
	return
}

func loadConfig(path, level string, jsonOut bool) (*config.Config, *slog.Logger, error) {
	cfg, err := config.Load(path)
	if err != nil {
		return nil, nil, err
	}
	if level == "" {
		level = cfg.Log.Level
	}
	var lvl slog.Level
	switch level {
	case "debug":
		lvl = slog.LevelDebug
	case "warn":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	opts := &slog.HandlerOptions{Level: lvl}
	var h slog.Handler
	if jsonOut || cfg.Log.Format == "json" {
		h = slog.NewJSONHandler(os.Stderr, opts)
	} else {
		h = slog.NewTextHandler(os.Stderr, opts)
	}
	return cfg, slog.New(h), nil
}

func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	configPath, logLevel, logJSON := commonFlags(fs)
	addr := fs.String("addr", "", "listen address (overrides config)")
	registryDir := fs.String("registry-dir", "data/registries", "token label registry directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, logger, err := loadConfig(*configPath, *logLevel, *logJSON)
	if err != nil {
		return err
	}
	if *addr != "" {
		cfg.Server.Addr = *addr
	}

	srv, err := httpapi.New(cfg, logger, *registryDir)
	if err != nil {
		return err
	}

	httpSrv := &http.Server{
		Addr:              cfg.Server.Addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: cfg.Server.ReadHeaderTimeout,
		ReadTimeout:       cfg.Server.ReadTimeout,
		WriteTimeout:      cfg.Server.WriteTimeout,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		logger.Info("hearim listening", "addr", cfg.Server.Addr,
			"public_endpoint", cfg.Gateway.PublicEndpoint)
		errCh <- httpSrv.ListenAndServe()
	}()

	select {
	case err := <-errCh:
		if err != nil && err != http.ErrServerClosed {
			return err
		}
	case <-ctx.Done():
		logger.Info("shutting down")
		shCtx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownGracePeriod)
		defer cancel()
		if err := httpSrv.Shutdown(shCtx); err != nil {
			return err
		}
		srv.Shutdown(shCtx)
	}
	return nil
}

func cmdProbe(args []string) error {
	fs := flag.NewFlagSet("probe", flag.ExitOnError)
	configPath, logLevel, logJSON := commonFlags(fs)
	providerID := fs.String("provider", "", "probe only this provider id")
	model := fs.String("model", "", "probe only this model (requires -provider)")
	out := fs.String("out", "", "write JSON report to file")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, logger, err := loadConfig(*configPath, *logLevel, *logJSON)
	if err != nil {
		return err
	}

	type target struct {
		provider config.ProviderConfig
		model    string
		modelCfg *config.ModelConfig
	}
	var targets []target
	for _, p := range cfg.Providers {
		if *providerID != "" && p.ID != *providerID {
			continue
		}
		var models []string
		if *model != "" {
			models = []string{*model}
		} else {
			models = p.Models.Names()
			if len(models) == 0 {
				// Alias-derived defaults.
				for alias, t := range cfg.ModelAliases {
					_ = alias
					if pid, m := splitTarget(t); pid == p.ID && m != "" {
						models = append(models, m)
					}
				}
			}
		}
		for _, m := range models {
			targets = append(targets, target{p, m, p.Models.Find(m)})
		}
	}
	if len(targets) == 0 {
		return fmt.Errorf("probe: no (provider, model) targets; configure providers[].models or pass -provider/-model")
	}

	reports := []*probe.Report{}
	for _, t := range targets {
		adpt, err := provider.NewAdapter(t.provider)
		if err != nil {
			return err
		}
		logger.Info("probing", "provider", t.provider.ID, "model", t.model)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		rep, err := probe.Run(ctx, adpt, t.model, cfg.Compiler, t.modelCfg)
		cancel()
		if err != nil {
			return err
		}
		fmt.Println(rep.Marshal())
		reports = append(reports, rep)
	}
	if *out != "" {
		data, _ := json.MarshalIndent(reports, "", "  ")
		if err := os.WriteFile(*out, data, 0o600); err != nil {
			return err
		}
	}
	// Exit non-zero when the Phase 0 gate fails for every target.
	anyGo := false
	for _, r := range reports {
		if r.Summary.Go {
			anyGo = true
		}
	}
	if !anyGo {
		return fmt.Errorf("probe: no target passed the go/no-go gate")
	}
	return nil
}

func cmdBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	configPath, logLevel, logJSON := commonFlags(fs)
	corpus := fs.String("corpus", "", "JSONL corpus path (required)")
	alias := fs.String("model", "", "public model alias (defaults to gateway.default_model)")
	permutations := fs.Int("permutations", 0, "label-order permutations per choice record (§12.3)")
	out := fs.String("out", "", "write JSON metrics to file")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *corpus == "" {
		return fmt.Errorf("bench: -corpus is required")
	}
	cfg, logger, err := loadConfig(*configPath, *logLevel, *logJSON)
	if err != nil {
		return err
	}
	f, err := os.Open(*corpus)
	if err != nil {
		return err
	}
	defer f.Close()
	recs, err := bench.LoadCorpus(readBufio(f))
	if err != nil {
		return err
	}
	if len(recs) == 0 {
		return fmt.Errorf("bench: empty corpus")
	}

	srv, err := httpapi.New(cfg, logger, "")
	if err != nil {
		return err
	}
	route, err := srv.Router.Resolve(*alias)
	if err != nil {
		return err
	}
	metrics, err := bench.Run(context.Background(), recs, route, srv.Compiler, cfg, *permutations)
	if err != nil {
		return err
	}
	data, _ := json.MarshalIndent(metrics, "", "  ")
	fmt.Println(string(data))
	if *out != "" {
		if err := os.WriteFile(*out, data, 0o600); err != nil {
			return err
		}
	}
	return nil
}

func splitTarget(t string) (string, string) {
	for i := 0; i < len(t); i++ {
		if t[i] == ':' {
			return t[:i], t[i+1:]
		}
	}
	return t, ""
}

func readBufio(f *os.File) *bufio.Reader { return bufio.NewReader(f) }
