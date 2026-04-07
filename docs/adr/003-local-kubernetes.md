# ADR-003: Local Kubernetes (k3d) Instead of Docker Compose or Virtual Machines

**Date**: March 2026
**Status**: Accepted  
**Deciders**: kmikol

## Context

The system is designed for learning production ML operations while running on a single developer laptop. Three deployment approaches were evaluated:

1. **Docker Compose**: Run all services (Postgres, Prometheus, Argo, etc.) as containers orchestrated by Compose
2. **Virtual Machines**: Create a realistic multi-machine setup using VirtualBox or Vagrant
3. **Local Kubernetes**: Use k3d (k3s in Docker) to run a lightweight Kubernetes cluster locally

## Decision

The system runs on **k3d**, a local Kubernetes cluster in Docker, deployed with Helm charts and Kubernetes manifests. All services (serving, monitoring, retraining, etc.) run as Kubernetes workloads.

Container images and Helm configurations are portable: the same YAML that runs on k3d will run on production Kubernetes clusters with minimal changes (update image registries, ingress hosts, storage classes).

## Rationale

**Environment parity**: Running on local Kubernetes means learning the exact tools, patterns, and configurations that production systems use:
- Pod resource limits, health checks, and lifecycle hooks behave the same
- Service discovery via DNS works identically
- StatefulSet management for databases is realistic
- Secret and ConfigMap handling matches production
- Ingress routing and traffic management is production-grade

**Skill transferability**: Engineers who build this system learn:
- How to write Kubernetes manifests
- Helm templating for environment-agnostic deployments
- kubectl debugging and cluster introspection
- RBAC for service accounts and role-based access
- StatefulSets and Deployments as distinct workload types

These skills directly transfer to production clusters (EKS, GKE, AKS) without retraining.

**Orchestration experience**: Unlike Docker Compose, Kubernetes handles:
- Service discovery across the cluster
- Rolling updates with health checks
- Resource allocation and QoS classes
- Networking policies and service-to-service communication
- Observability hooks (liveness/readiness probes)

Learning Kubernetes locally builds confidence before touching production infrastructure.

**Tooling ecosystem**: k3d enables standard tools:
- `kubectl` for troubleshooting and inspection
- `helm` for templating and versioning deployments
- `argo` CLI for workflow submission and debugging
- Standard Kubernetes dashboards and monitoring

This isn't possible (or is cumbersome) with Docker Compose.

**Cost of entry**: k3d has minimal overhead:
- Runs in Docker (no additional VMs)
- Fast startup (seconds, not minutes)
- Uses ~2GB RAM for a full cluster
- No licensing or special infrastructure needed

## Consequences

**Positive**:
- Deployments are portable to production Kubernetes almost unchanged
- Learning realistic infrastructure patterns
- Full Kubernetes feature set available (StatefulSets, DaemonSets, Jobs, CronJobs) for future expansion
- Standard debugging tools (kubectl logs, describe, exec, port-forward)
- Integrates naturally with Argo Workflows and Argo Rollouts (both are Kubernetes-native)

**Negative**:
- More initial learning curve than Docker Compose (Kubernetes concepts like Deployments, Services, Ingress)
- Slower container startup times vs. Compose (orchestration overhead, health checks)
- Local networking limitations: no true multi-machine simulation
- Latency characteristics don't match production (local Docker networking vs. real cluster networking)
- Network partition simulation is artificial (hard to test failure recovery realistically)

**Resource constraints**:
- Limited resources on a laptop (typically 4-8 cores, 8-16GB RAM)
- Cannot simulate true geographic distribution or high-availability scenarios
- Pod density and node pressure handling is artificial

## Limitations and Gaps

**Latency**: Local Docker networking is much faster than real clusters. Inter-service latency is sub-millisecond; production latency is typically 10-100ms depending on network and geography. This means performance testing on k3d is unreliable.

**Network failures**: Real clusters experience packet loss, latency spikes, and temporary partition. k3d with Docker networking is very reliable, so failure recovery code isn't thoroughly tested.

**Scale**: k3d can run ~10-15 pods comfortably on a laptop. Production scenarios with thousands of replicas cannot be simulated.

**Observability**: k3d doesn't capture real cluster metrics like node memory pressure or kubelet eviction. This means testing graceful shutdown under resource pressure is limited.

## Alternatives Considered

### Alternative 1: Docker Compose
Use Docker Compose to orchestrate all containers locally.

**Pros**: Simpler to learn, faster startup, less resource overhead, fewer moving parts  
**Cons**: Not representative of production infrastructure; doesn't teach Kubernetes concepts; no service discovery or sophisticated health checks; limited debugging tools; not portable to Kubernetes environments

### Alternative 2: Virtual Machines (Vagrant/VirtualBox)
Create a multi-machine cluster locally with real VMs.

**Pros**: Realistic network behavior, can simulate multi-node failures, closer to production networking  
**Cons**: High resource overhead (need multiple VMs), slow startup, complex setup, harder to debug, not practical on a laptop

### Alternative 3: Managed Cloud (GKE/EKS free tier)
Use a real cloud Kubernetes cluster with free tier benefits.

**Pros**: Real production environment, realistic latency and networking, full feature parity, no local resource constraints  
**Cons**: Not "local" (requires internet, cloud account), adds cost (even free tier has egress charges), less suitable for learning (can't freely break and rebuild), introduces cloud-specific features

## Mitigations for Limitations

1. **Teach latency behavior separately**: Document that local latencies are unrealistic; recommend reading about production cluster networking
2. **Failure injection tools**: Use Chaos Mesh or Kyverno to simulate failures on k3d for testing resilience (not fully equivalent, but better)
3. **Scale testing elsewhere**: For high-scale scenarios, developers should move to a real cluster to validate behavior
4. **Integration tests in CI/CD**: Automated tests should run on both k3d and a staging cluster to catch environment-specific issues early

## Related Decisions

- **ADR-001**: Event-driven retraining relies on Kubernetes CronJobs and Workflows (Argo), which require K8s
- **ADR-002**: Canary rollouts use Argo Rollouts, which is a Kubernetes operator

## Future Considerations

1. **Multi-node simulation**: If real cluster networking becomes critical, explore kind (Kubernetes in Docker) with multiple nodes instead of k3d
2. **Network simulation**: Integrate Chaos Mesh to inject latency/packet loss and test failure recovery
3. **Resource limits testing**: Use resource quotas and limit ranges to simulate resource pressure scenarios
4. **Production rehearsal**: Periodic validation by deploying to staging Kubernetes cluster and comparing behavior

## Implementation Notes

- k3d clusters are created with `k3d cluster create ml-system --kubeconfig-update-default`
- Helm deployments via `helm install ml-system helm/ml-system/ -f helm/ml-system/values-local.yaml`
- All manifests in `argo/` are standard Kubernetes YAML, no k3d-specific extensions
- Services communicate via Kubernetes DNS (e.g., `prometheus.ml-system.svc.cluster.local`)
