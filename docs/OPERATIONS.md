# EL (Eager Learner) – Operational Runbook

_Last updated: 2025-07-27_

## 1. Runtime Overview

| Component | Runtime | Scaling |
|-----------|---------|---------|
| FastAPI API | Gunicorn + Uvicorn workers | CPU-bound: scale horizontally via ECS/Fargate task count |
| Neo4j Aura | Managed | Auto |
| Redis | AWS ElastiCache | Multi-AZ |
| Prometheus | Grafana Cloud Agent | Static job scraping `/metrics` |

## 2. Environment Variables

```
OPENAI_API_KEY=...
AUTH0_DOMAIN=...
AUTH0_AUDIENCE=...
NEO4J_URI=bolt://...
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
REDIS_URL=redis://...
AUTO_INGEST_NEO4J=1
PUBLISH_CONTEXT_STREAM=1
```

## 3. Deploy Procedure

```bash
# 1. Build & push (handled by CI on main)
# 2. Update image tag in Helm values
helm upgrade el-agent charts/el-agent \
  --set image.tag=<sha>
# 3. ArgoCD sync (auto) – verify rollout
```

## 4. Health Checks

* **Liveness**: `GET /healthz` (FastAPI default root returns 200)  
* **Readiness**: `GET /extract` with small dummy payload (JWT required)
* **Metrics**: `GET /metrics` – Prometheus exposition format

## 5. Alerting Rules (Prometheus)

| Alert | Expression | Severity |
|-------|------------|----------|
| High error rate | `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05` | critical |
| Slow pipeline | `histogram_quantile(0.95, sum(rate(request_duration_seconds_bucket[5m])) by (le)) > 8` | warning |
| Token surge | `increase(el_openai_tokens_total[1h]) > 5e5` | warning |

## 6. Backup & Restore

* **Neo4j Aura**: automated daily backups (7-day retention) – restore via Aura console.  
* **LLM Logs**: `logs/llm_calls.jsonl` uploaded to S3 every 4 h via sidecar.

## 7. Security Checklist

- [x] JWT verification (`RS256`)  
- [x] TLS enforced via ALB (HTTPS)  
- [x] Secrets stored in AWS Secrets Manager  
- [x] Image scanning through Trivy in CI

## 8. Troubleshooting

| Symptom | Possible Cause | Mitigation |
|---------|----------------|-----------|
| 401 Unauthorized | Invalid/missing JWT | Verify Auth0 token & audience |
| 500 from `/extract` | OpenAI error / schema mismatch | Check CloudWatch logs, retry with temp=0 |
| High latency | LLM throttling | Ensure rate-limit compliance, consider caching |

---

For detailed architecture and pipeline description, see `IMPLEMENTATION_PLAN.md`. 