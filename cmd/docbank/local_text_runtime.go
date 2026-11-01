package main

import (
	"context"
	"encoding/json/v2"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"

	"go.kenn.io/docbank/internal/config"
	"go.kenn.io/docbank/internal/processing"
	"go.kenn.io/kit/atomicfile"
	"go.kenn.io/kit/embedlocal"
)

func readLocalText(root string) (config.LocalTextConfig, error) {
	var choice config.LocalTextConfig
	data, err := os.ReadFile(filepath.Join(root, "local-text.json"))
	if errors.Is(err, os.ErrNotExist) {
		return choice, nil
	}
	if err == nil {
		err = json.Unmarshal(data, &choice, json.RejectUnknownMembers(true))
		if err == nil && choice.Version != 1 {
			err = fmt.Errorf("unsupported local-text format version %d", choice.Version)
		}
		if err == nil && (choice.Profile == "" || (choice.Server != "auto" && choice.Server != "local")) {
			err = errors.New("local-text selection requires a profile and auto or local server")
		}
	}
	return choice, err
}

func saveLocalText(root string, choice config.LocalTextConfig) error {
	choice.Version = 1
	data, err := json.Marshal(choice)
	if err != nil {
		return err
	}
	return atomicfile.WriteFile(filepath.Join(root, "local-text.json"), data)
}

// Only this isolated view rewrites embedding definitions. Its sole processing
// profile keeps its public name and binding names, including shared bindings.
func localTextCandidate(cfg config.Config, name, mode string) (config.Config, bool, bool, error) {
	if mode != "auto" && mode != "local" {
		return cfg, false, false, processing.ErrProfileNotConfigured
	}
	cfg.EmbeddingProfiles = maps.Clone(cfg.EmbeddingProfiles)
	cfg.RenditionProfiles = maps.Clone(cfg.RenditionProfiles)
	cfg.RetrievalProfiles = maps.Clone(cfg.RetrievalProfiles)
	cfg.ProcessingProfiles = maps.Clone(cfg.ProcessingProfiles)
	if name == config.LocalTextProfile {
		if _, exists := cfg.ProcessingProfiles[name]; !exists {
			var err error
			cfg, err = cfg.WithLocalText()
			if err != nil {
				return cfg, false, false, err
			}
		}
	}
	profile, ok := cfg.ProcessingProfiles[name]
	if !ok || len(profile.Embeddings) == 0 {
		return cfg, false, false, processing.ErrProfileNotConfigured
	}
	fallback, err := config.Default().WithLocalText()
	if err != nil {
		return cfg, false, false, err
	}
	managed, rebuild := false, false
	textBindings := 0
	for _, binding := range profile.Embeddings {
		original, err := cfg.EmbeddingProfile(binding)
		if err != nil {
			return cfg, false, false, err
		}
		if original.InputKind != "rendition_chunk" {
			continue
		}
		textBindings++
		if original.Runtime == nil {
			return cfg, false, false, processing.ErrProfileNotConfigured
		}
		if mode == "auto" && original.Runtime.Endpoint != "" {
			continue
		}
		next := original
		if original.Model != "nomic-embed-text" {
			next = fallback.EmbeddingProfiles[config.LocalTextProfile]
			next.Activation = original.Activation
			rebuild = true
		}
		if next.Runtime.AdapterContract != "docbank-openai-compatible-embeddings/v1" {
			return cfg, false, false, fmt.Errorf("%w: profile %q requires an OpenAI-compatible Nomic runtime; choose --profile local-text", processing.ErrLocalTextConfiguration, name)
		}
		if next.Dimensions != 768 {
			return cfg, false, false, fmt.Errorf("%w: profile %q requires Nomic with 768 dimensions; choose --profile local-text or configure 768 dimensions", processing.ErrLocalTextConfiguration, name)
		}
		next.Runtime = new(*next.Runtime)
		next.Runtime.Managed, next.Runtime.ServerMode = true, mode
		next.Runtime.ProviderEgressConfig = config.ProviderEgressConfig{}
		next.Runtime.ProviderRevisionHeader = ""
		next.Runtime.DeploymentEpoch = next.Runtime.ModelRevision
		next.Embedder = nil
		cfg.EmbeddingProfiles[binding] = next
		managed = true
	}
	if textBindings == 0 {
		return cfg, false, false, processing.ErrProfileNotConfigured
	}
	cfg.ProcessingProfiles = map[string]config.ProcessingProfileConfig{name: profile}
	return cfg, managed, rebuild, nil
}

func defaultLocalTextProfile(cfg config.Config) string {
	for _, name := range slices.Sorted(maps.Keys(cfg.ProcessingProfiles)) {
		p := cfg.ProcessingProfiles[name]
		for _, binding := range p.Embeddings {
			e, err := cfg.EmbeddingProfile(binding)
			if err == nil && e.Runtime != nil && e.InputKind == "rendition_chunk" && e.Model == "nomic-embed-text" && e.Dimensions == 768 {
				if _, _, _, err := localTextCandidate(cfg, name, "local"); err == nil {
					return name
				}
			}
		}
	}
	return config.LocalTextProfile
}

func localTextStartup(ctx context.Context, managed bool, mode string) error {
	if !managed {
		return nil
	}
	_, err := embedlocal.Endpoint(ctx, embedlocal.Options{SkipOllama: mode == "local"})
	if err != nil {
		return fmt.Errorf("embeddings server is not ready; run docbank processing local to install and start it: %w", err)
	}
	return nil
}

func projectLocalTextProfile(cfg config.Config, profiles map[string]processing.ProfileConfig, candidate config.Config, name string, selected processing.ProfileConfig) (config.Config, error) {
	// Registries share identical descriptors, so retain unchanged provider objects.
	for _, existing := range profiles {
		if selected.RenditionProvider != nil && existing.RenditionProvider != nil && selected.RenditionProvider.Descriptor().Fingerprint == existing.RenditionProvider.Descriptor().Fingerprint {
			selected.RenditionProvider = existing.RenditionProvider
		}
		for binding, provider := range selected.EmbeddingProviders {
			for _, original := range existing.EmbeddingProviders {
				if provider.Descriptor().Fingerprint == original.Descriptor().Fingerprint {
					selected.EmbeddingProviders[binding] = original
				}
			}
		}
	}
	resolved, err := candidate.ProcessingProfile(name)
	if err != nil {
		return cfg, err
	}
	profiles[name] = selected
	cfg.EffectiveProfiles = map[string]config.ResolvedProcessingProfile{name: resolved}
	cfg.ProcessingProfiles = maps.Clone(cfg.ProcessingProfiles)
	cfg.ProcessingProfiles[name] = candidate.ProcessingProfiles[name]
	return cfg, nil
}
