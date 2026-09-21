//go:build !unix

package selfupdate

import "os"

// On platforms without a /dev/stderr char device, fall back to checking the
// stderr file descriptor's mode directly.
func stderrIsTerminal() bool {
	fi, err := os.Stderr.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}
