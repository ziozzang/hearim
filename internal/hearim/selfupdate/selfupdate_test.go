package selfupdate

import "testing"

func TestCompareVersions(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"0.1.0", "0.2.0", -1},
		{"v0.2.0", "0.2.0", 0},
		{"0.10.0", "0.9.9", 1},
		{"1.0", "1.0.0", 0},
		{"0.2.0-rc1", "0.2.0", 0}, // pre-release suffixes ignored
	}
	for _, c := range cases {
		if got := CompareVersions(c.a, c.b); got != c.want {
			t.Errorf("CompareVersions(%q,%q) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}

func TestAssetName(t *testing.T) {
	n, err := AssetName("0.2.0", "linux", "amd64")
	if err != nil || n != "hearim_0.2.0_linux_x86_64" {
		t.Errorf("asset = %q %v", n, err)
	}
	n, err = AssetName("0.2.0", "windows", "arm64")
	if err != nil || n != "hearim_0.2.0_windows_arm64.exe" {
		t.Errorf("asset = %q %v", n, err)
	}
	if _, err := AssetName("0.2.0", "plan9", "amd64"); err == nil {
		t.Error("unsupported platform should error")
	}
}

func TestReleaseFindAsset(t *testing.T) {
	r := &Release{TagName: "v1", Assets: []Asset{{Name: "SHA256SUMS"}, {Name: "hearim_1.0.0_linux_x86_64"}}}
	if _, ok := r.FindAsset("missing"); ok {
		t.Error("missing asset found")
	}
	if a, ok := r.FindAsset("SHA256SUMS"); !ok || a.Name != "SHA256SUMS" {
		t.Error("SHA256SUMS not found")
	}
	if r.Version() != "1" {
		t.Errorf("version = %q", r.Version())
	}
}
