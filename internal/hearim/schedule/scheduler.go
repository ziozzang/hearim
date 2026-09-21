// Package schedule implements the engine-aware, prefix-affine scheduler of
// TODO.md §8.3: bounded concurrency lanes per provider, micro-batch grouping
// of same-prefix questions, and warm-prefix preference.
//
// The gateway scheduler does not replace engine-internal batching; it only
// picks replica-side ordering and keeps same-prefix requests temporally
// adjacent so provider prompt caches survive across the fan-out.
package schedule

import (
	"container/list"
	"context"
	"errors"
	"sync"
	"time"
)

// Task is one unit of upstream work, tagged with its prefix cache key.
type Task struct {
	PrefixKey string
	Run       func(ctx context.Context)
}

// ErrClosed is returned after Shutdown.
var ErrClosed = errors.New("schedule: scheduler closed")

// Scheduler executes tasks on bounded lanes with prefix affinity.
type Scheduler struct {
	maxLanes int
	window   time.Duration
	mu       sync.Mutex
	cond     *sync.Cond
	queue    *list.List     // *queuedTask
	running  map[string]int // prefixKey -> active lanes
	warm     map[string]time.Time
	closed   bool
	wg       sync.WaitGroup
}

type queuedTask struct {
	task     Task
	enqueued time.Time
	cancel   context.CancelFunc
	dead     bool
}

// New creates a scheduler with the given lane count and micro-batch window
// (TODO.md §8.3 suggests 5-15ms, tuned by measurement).
func New(lanes int, window time.Duration) *Scheduler {
	if lanes < 1 {
		lanes = 1
	}
	if window <= 0 {
		window = 10 * time.Millisecond
	}
	s := &Scheduler{
		maxLanes: lanes,
		window:   window,
		queue:    list.New(),
		running:  map[string]int{},
		warm:     map[string]time.Time{},
	}
	s.cond = sync.NewCond(&s.mu)
	for i := 0; i < lanes; i++ {
		s.wg.Add(1)
		go s.worker()
	}
	return s
}

// Submit enqueues a task and blocks until it completes (or ctx ends, which
// cancels the task's context and abandons waiting).
func (s *Scheduler) Submit(ctx context.Context, t Task) error {
	runCtx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	inner := t.Run
	t.Run = func(c context.Context) {
		defer close(done)
		if inner != nil {
			inner(runCtx)
		}
	}
	qt := &queuedTask{task: t, enqueued: time.Now(), cancel: cancel}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		cancel()
		return ErrClosed
	}
	s.queue.PushBack(qt)
	s.mu.Unlock()
	s.cond.Broadcast()

	select {
	case <-done:
		cancel()
		return nil
	case <-ctx.Done():
		// Mark the task dead so a worker skips it if still queued; a
		// running task observes runCtx cancellation.
		s.mu.Lock()
		qt.dead = true
		s.mu.Unlock()
		s.cond.Broadcast()
		return ctx.Err()
	}
}

// Shutdown stops accepting tasks and waits for in-flight work.
func (s *Scheduler) Shutdown(ctx context.Context) error {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	s.cond.Broadcast()
	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *Scheduler) worker() {
	defer s.wg.Done()
	for {
		s.mu.Lock()
		for s.queue.Len() == 0 && !s.closed {
			s.cond.Wait()
		}
		if s.queue.Len() == 0 && s.closed {
			s.mu.Unlock()
			return
		}
		qt := s.pickLocked()
		if qt == nil {
			s.mu.Unlock()
			continue
		}
		prefix := qt.task.PrefixKey
		if prefix != "" {
			s.running[prefix]++
		}
		s.mu.Unlock()

		if !qt.dead {
			qt.task.Run(context.Background())
		}

		s.mu.Lock()
		if prefix != "" {
			s.running[prefix]--
			if s.running[prefix] == 0 {
				delete(s.running, prefix)
			}
			if !qt.dead {
				s.warm[prefix] = time.Now()
			}
			s.pruneWarmLocked()
		}
		s.mu.Unlock()
		s.cond.Broadcast()
	}
}

// pickLocked selects the next live task. Priority (TODO.md §8.3):
//  1. same prefix as currently running work
//  2. recently warm prefix (recency order)
//  3. oldest queued
//
// Dead (cancelled) tasks are removed opportunistically.
func (s *Scheduler) pickLocked() *queuedTask {
	for el := s.queue.Front(); el != nil; {
		next := el.Next()
		if el.Value.(*queuedTask).dead {
			s.queue.Remove(el)
		}
		el = next
	}
	if s.queue.Len() == 0 {
		return nil
	}
	for _, prefix := range s.runningPrefixesLocked() {
		if qt := s.takePrefixLocked(prefix); qt != nil {
			return qt
		}
	}
	for _, prefix := range s.warmPrefixesByRecencyLocked() {
		if qt := s.takePrefixLocked(prefix); qt != nil {
			return qt
		}
	}
	el := s.queue.Front()
	if el == nil {
		return nil
	}
	return s.queue.Remove(el).(*queuedTask)
}

func (s *Scheduler) runningPrefixesLocked() []string {
	out := make([]string, 0, len(s.running))
	for p := range s.running {
		out = append(out, p)
	}
	return out
}

func (s *Scheduler) warmPrefixesByRecencyLocked() []string {
	type pe struct {
		p  string
		at time.Time
	}
	var entries []pe
	for p, at := range s.warm {
		if time.Since(at) <= 10*time.Second {
			entries = append(entries, pe{p, at})
		}
	}
	for i := 1; i < len(entries); i++ {
		for j := i; j > 0 && entries[j].at.After(entries[j-1].at); j-- {
			entries[j], entries[j-1] = entries[j-1], entries[j]
		}
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		out = append(out, e.p)
	}
	return out
}

func (s *Scheduler) takePrefixLocked(prefix string) *queuedTask {
	for el := s.queue.Front(); el != nil; el = el.Next() {
		qt := el.Value.(*queuedTask)
		if qt.task.PrefixKey == prefix && !qt.dead {
			s.queue.Remove(el)
			return qt
		}
	}
	return nil
}

func (s *Scheduler) pruneWarmLocked() {
	if len(s.warm) <= 1024 {
		return
	}
	cutoff := time.Now().Add(-10 * time.Second)
	for p, at := range s.warm {
		if at.Before(cutoff) {
			delete(s.warm, p)
		}
	}
}

// Lanes reports the configured concurrency bound.
func (s *Scheduler) Lanes() int { return s.maxLanes }

// Window reports the micro-batch window.
func (s *Scheduler) Window() time.Duration { return s.window }
