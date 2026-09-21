// Package selfupdate replaces the running hearim binary with the latest
// GitHub release build. The design follows ziozzang/hftools
// internal/selfupdate: stdlib-only and integrity-first — every downloaded
// binary is checked against the SHA-256 recorded in the release's SHA256SUMS
// asset before it may replace the current executable.
package selfupdate

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// DefaultAPIBase is the GitHub REST API root.
const DefaultAPIBase = "https://api.github.com"

// Release is the subset of a GitHub release we consume.
type Release struct {
	TagName string  `json:"tag_name"`
	Name    string  `json:"name"`
	Body    string  `json:"body"`
	Assets  []Asset `json:"assets"`
}

// Asset is one downloadable release artifact.
type Asset struct {
	Name string `json:"name"`
	URL  string `json:"browser_download_url"`
	Size int64  `json:"size"`
}

// FindAsset returns the asset with the given name.
func (r *Release) FindAsset(name string) (Asset, bool) {
	for _, a := range r.Assets {
		if a.Name == name {
			return a, true
		}
	}
	return Asset{}, false
}

// Version strips a leading "v" from the release tag.
func (r *Release) Version() string { return strings.TrimPrefix(r.TagName, "v") }

func newRequest(ctx context.Context, method, url, token string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", "hearim-selfupdate")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return req, nil
}

// LatestRelease fetches the newest published release for owner/repo. token
// may be empty; when set it raises the unauthenticated rate limit.
func LatestRelease(ctx context.Context, client *http.Client, apiBase, repo, token string) (*Release, error) {
	if apiBase == "" {
		apiBase = DefaultAPIBase
	}
	resp, err := doReleaseRequest(ctx, client, apiBase+"/repos/"+repo+"/releases/latest", token)
	if err != nil {
		return nil, err
	}
	return decodeRelease(resp)
}

// ReleaseByTag fetches a specific release by its tag name.
func ReleaseByTag(ctx context.Context, client *http.Client, apiBase, repo, tag, token string) (*Release, error) {
	if apiBase == "" {
		apiBase = DefaultAPIBase
	}
	resp, err := doReleaseRequest(ctx, client, apiBase+"/repos/"+repo+"/releases/tags/"+tag, token)
	if err != nil {
		return nil, err
	}
	return decodeRelease(resp)
}

func doReleaseRequest(ctx context.Context, client *http.Client, url, token string) (*http.Response, error) {
	req, err := newRequest(ctx, http.MethodGet, url, token)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<10))
		return nil, fmt.Errorf("GitHub API %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	return resp, nil
}

func decodeRelease(resp *http.Response) (*Release, error) {
	defer resp.Body.Close()
	var rel Release
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&rel); err != nil {
		return nil, fmt.Errorf("decode release: %w", err)
	}
	if rel.TagName == "" {
		return nil, fmt.Errorf("release has no tag")
	}
	return &rel, nil
}

// AssetName returns the release artifact name for a platform, matching the
// scheme produced by scripts/build-release.sh:
//
//	hearim_<version>_<platform>_<arch>[.exe]
func AssetName(version, goos, goarch string) (string, error) {
	platform := map[string]string{"darwin": "macos", "windows": "windows", "linux": "linux"}[goos]
	arch := map[string]string{"amd64": "x86_64", "arm64": "arm64"}[goarch]
	if platform == "" || arch == "" {
		return "", fmt.Errorf("no release build for %s/%s", goos, goarch)
	}
	name := fmt.Sprintf("hearim_%s_%s_%s", version, platform, arch)
	if goos == "windows" {
		name += ".exe"
	}
	return name, nil
}

// CurrentAssetName returns AssetName for the running platform.
func CurrentAssetName(version string) (string, error) {
	return AssetName(version, runtime.GOOS, runtime.GOARCH)
}

// CompareVersions compares dotted numeric versions (leading "v" and any
// pre-release suffix after "-" or "+" are ignored). It returns -1 if a < b,
// 0 if equal, and +1 if a > b.
func CompareVersions(a, b string) int {
	an := versionFields(a)
	bn := versionFields(b)
	for i := 0; i < len(an) || i < len(bn); i++ {
		var x, y int
		if i < len(an) {
			x = an[i]
		}
		if i < len(bn) {
			y = bn[i]
		}
		if x != y {
			if x < y {
				return -1
			}
			return 1
		}
	}
	return 0
}

func versionFields(v string) []int {
	v = strings.TrimPrefix(strings.TrimSpace(v), "v")
	if i := strings.IndexAny(v, "-+"); i >= 0 {
		v = v[:i]
	}
	parts := strings.Split(v, ".")
	out := make([]int, len(parts))
	for i, p := range parts {
		n, _ := strconv.Atoi(p)
		out[i] = n
	}
	return out
}

// Checksums downloads and parses the release's SHA256SUMS asset into a map
// of artifact name -> lowercase hex SHA-256.
func Checksums(ctx context.Context, client *http.Client, rel *Release, token string) (map[string]string, error) {
	asset, ok := rel.FindAsset("SHA256SUMS")
	if !ok {
		return nil, fmt.Errorf("release %s has no SHA256SUMS asset", rel.TagName)
	}
	body, err := download(ctx, client, asset.URL, 1<<20, token)
	if err != nil {
		return nil, err
	}
	sums := map[string]string{}
	for _, line := range strings.Split(string(body), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 {
			sums[fields[1]] = strings.ToLower(fields[0])
		}
	}
	return sums, nil
}

// DownloadVerified fetches asset into a temp file in dir, checks its SHA-256
// against wantSHA, and returns the temp file path. The caller renames it
// into place (or removes it). On any error the temp file is cleaned up.
func DownloadVerified(ctx context.Context, client *http.Client, asset Asset, wantSHA, dir, token string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, asset.URL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "hearim-selfupdate")
	// Release asset downloads on private repositories need the token too
	// (browser_download_url is not pre-signed there).
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: %s", asset.Name, resp.Status)
	}
	tmp, err := os.CreateTemp(dir, ".hearim-update-*")
	if err != nil {
		return "", err
	}
	tmpName := tmp.Name()
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(tmp, h), resp.Body); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return "", err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return "", err
	}
	got := hex.EncodeToString(h.Sum(nil))
	if wantSHA != "" && !strings.EqualFold(got, wantSHA) {
		_ = os.Remove(tmpName)
		return "", fmt.Errorf("checksum mismatch for %s: got %s, want %s", asset.Name, got, wantSHA)
	}
	if err := os.Chmod(tmpName, 0o755); err != nil {
		_ = os.Remove(tmpName)
		return "", err
	}
	return tmpName, nil
}

// ReplaceExecutable moves src onto dest, replacing the (possibly running)
// executable. On Unix this is an atomic rename; on Windows the running
// binary is moved aside first because it cannot be overwritten in place.
func ReplaceExecutable(dest, src string) error {
	if runtime.GOOS == "windows" {
		old := dest + ".old"
		_ = os.Remove(old)
		if err := os.Rename(dest, old); err != nil {
			return err
		}
		if err := os.Rename(src, dest); err != nil {
			// Roll back so the tool is not left without a binary.
			_ = os.Rename(old, dest)
			return err
		}
		_ = os.Remove(old) // best effort; the running .old may still be locked
		return nil
	}
	return os.Rename(src, dest)
}

// ResolveExecutable returns the real path of the running binary, following
// symlinks so the underlying file (not a symlink) is the one replaced.
func ResolveExecutable() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	if resolved, err := filepath.EvalSymlinks(exe); err == nil {
		return resolved, nil
	}
	return exe, nil
}

func download(ctx context.Context, client *http.Client, url string, limit int64, token string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "hearim-selfupdate")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download %s: %s", url, resp.Status)
	}
	return io.ReadAll(io.LimitReader(resp.Body, limit))
}
