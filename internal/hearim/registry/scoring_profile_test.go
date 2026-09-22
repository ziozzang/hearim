package registry

import "testing"

func TestScoringSettingsInvalidateRegistry(t *testing.T) {
	opts := Options{BackendModel: "m", Endpoint: "completions", ScoringProfile: "raw-profile", DelimiterCandidates: []string{"\n"}}
	r := Registry{BackendModel: opts.BackendModel, Endpoint: opts.Endpoint, ScoringProfile: opts.ScoringProfile, Delimiter: "\n"}
	if OptionsKey(opts) != r.Key() {
		t.Fatal("lookup/build key mismatch")
	}
	old := r.Key()
	r.ScoringProfile = "mask-disabled-profile"
	if old == r.Key() {
		t.Fatal("profile change reused stale registry")
	}
}
