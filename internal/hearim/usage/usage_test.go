package usage

import (
	"testing"

	"hearim/internal/hearim/config"
)

func TestBudgetTiers(t *testing.T) {
	b := NewBudget(config.BudgetConfig{IncludedUSD: 60, WarnAt: 0.7, ThrottleAt: 0.9, HardStopAt: 1.0})
	b.Record(10)
	if s, f := b.State(); s != BudgetOK || f > 0.17 {
		t.Errorf("state = %v %v", s, f)
	}
	b.Record(32) // 42/60 = 0.7
	if s, _ := b.State(); s != BudgetWarn {
		t.Errorf("state = %v, want warn", s)
	}
	b.Record(12) // 54/60 = 0.9
	if s, _ := b.State(); s != BudgetThrottle {
		t.Errorf("state = %v, want throttle", s)
	}
	b.Record(6) // 60/60 = 1.0
	if s, _ := b.State(); s != BudgetHardStop {
		t.Errorf("state = %v, want hard stop", s)
	}
}

func TestLedgerSeparatesReportedAndEstimated(t *testing.T) {
	l := NewLedger()
	l.Record("route-a", Aggregate{
		InputTokens: 1000, OutputTokens: 3,
		ReportedCachedTokens: 600, EstimatedCachedTokens: 700,
	}, 1.5, 0.012)
	row := l.Snapshot()["route-a"]
	if row.UncachedPrefillTokens != 400 || row.CachedPrefillTokens != 600 {
		t.Errorf("row = %+v", row)
	}
	if row.DecodeTokens != 3 || row.USD != 0.012 {
		t.Errorf("row = %+v", row)
	}
}
