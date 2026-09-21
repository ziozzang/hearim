package schedule

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestConcurrencyBounded(t *testing.T) {
	var running, maxSeen int64
	var mu sync.Mutex
	s := New(3, time.Millisecond)
	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = s.Submit(context.Background(), Task{PrefixKey: "p", Run: func(ctx context.Context) {
				cur := atomic.AddInt64(&running, 1)
				mu.Lock()
				if cur > maxSeen {
					maxSeen = cur
				}
				mu.Unlock()
				time.Sleep(5 * time.Millisecond)
				atomic.AddInt64(&running, -1)
			}})
		}()
	}
	wg.Wait()
	_ = s.Shutdown(context.Background())
	if maxSeen > 3 {
		t.Errorf("max concurrent = %d, lanes = 3", maxSeen)
	}
}

func TestPrefixAffinity(t *testing.T) {
	// Same-prefix tasks should be grouped adjacently in time: while a prefix
	// is running, a queued task with the same prefix beats an older task
	// with a different prefix.
	var mu sync.Mutex
	order := []string{}
	release := make(chan struct{})

	s := New(1, time.Millisecond)
	// First task blocks until released.
	go func() {
		_ = s.Submit(context.Background(), Task{PrefixKey: "A", Run: func(ctx context.Context) {
			mu.Lock()
			order = append(order, "A1")
			mu.Unlock()
			<-release
		}})
	}()
	time.Sleep(20 * time.Millisecond) // let A1 start

	// B enqueued first (older), A2 second.
	go func() {
		_ = s.Submit(context.Background(), Task{PrefixKey: "B", Run: func(ctx context.Context) {
			mu.Lock()
			order = append(order, "B")
			mu.Unlock()
		}})
	}()
	time.Sleep(10 * time.Millisecond)
	go func() {
		_ = s.Submit(context.Background(), Task{PrefixKey: "A", Run: func(ctx context.Context) {
			mu.Lock()
			order = append(order, "A2")
			mu.Unlock()
		}})
	}()
	time.Sleep(10 * time.Millisecond)
	close(release)
	_ = s.Shutdown(context.Background())

	mu.Lock()
	defer mu.Unlock()
	if len(order) != 3 {
		t.Fatalf("order = %v", order)
	}
	if order[1] != "A2" {
		t.Errorf("prefix affinity failed: %v (A2 should run before B)", order)
	}
}

func TestSubmitRespectsContext(t *testing.T) {
	s := New(1, time.Millisecond)
	defer func() { _ = s.Shutdown(context.Background()) }()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var ran int64
	err := s.Submit(ctx, Task{PrefixKey: "x", Run: func(ctx context.Context) {
		atomic.AddInt64(&ran, 1)
	}})
	if err == nil {
		t.Error("cancelled context should return early")
	}
	// The dead task must be skipped, not executed.
	time.Sleep(50 * time.Millisecond)
	if atomic.LoadInt64(&ran) != 0 {
		t.Error("cancelled task should not run")
	}
}
