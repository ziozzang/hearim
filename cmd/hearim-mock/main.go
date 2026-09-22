// hearim-mock is a local, model-free inference protocol scenario server.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"hearim/internal/mockinfer"
)

func main() {
	engine := flag.String("engine", "vllm", "vllm | sglang | llama.cpp")
	scenario := flag.String("scenario", "normal", "scenario: "+strings.Join(mockinfer.Scenarios, ", "))
	addr := flag.String("addr", "127.0.0.1:18080", "listen address")
	models := flag.String("models", "", "optional model=scenario pairs separated by commas (native llama.cpp is single-model)")
	flag.Parse()
	profiles := map[string]string{}
	if *models != "" {
		for _, pair := range strings.Split(*models, ",") {
			parts := strings.SplitN(pair, "=", 2)
			if len(parts) != 2 || parts[0] == "" {
				log.Fatal("models must be model=scenario pairs")
			}
			profiles[parts[0]] = parts[1]
		}
	}
	mock, err := mockinfer.New(mockinfer.Options{Engine: *engine, Scenario: *scenario, ModelScenarios: profiles})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Synthetic %s mock listening on http://%s (scenario=%s); no model weights loaded\n", *engine, *addr, *scenario)
	srv := &http.Server{Addr: *addr, Handler: mock, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 15 * time.Second, WriteTimeout: 15 * time.Second}
	log.Fatal(srv.ListenAndServe())
}
