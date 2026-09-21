//go:build unix

package selfupdate

import "os"

func stderrIsTerminal() bool {
	fi, err := os.Stat("/dev/stderr")
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}
