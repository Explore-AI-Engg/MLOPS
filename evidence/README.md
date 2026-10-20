# Submission evidence checklist

Use this directory for screenshots captured from **this repository and this student's own workflow**. Do not copy screenshots from another assignment.

Recommended filenames:

1. `01-data-preparation.png`
2. `02-eda-feature-distributions.png`
3. `03-eda-correlation-heatmap.png`
4. `04-eda-class-balance.png`
5. `05-mlflow-run.png`
6. `06-tests-and-lint.png`
7. `07-docker-build.png`
8. `08-local-health.png`
9. `09-local-predict.png`
10. `10-ci-success.png`
11. `11-kubernetes-rollout-success.png`
12. `12-public-health.png`
13. `13-public-predict.png`
14. `14-public-metrics.png`

Items 11–14 are now verified against the public service at `http://104.154.36.8:8000`. A screenshot of `CrashLoopBackOff`, HTTP 503, or a timed-out rollout may be kept as troubleshooting evidence, but must be labelled as a failure and must not be used as proof of successful deployment.

Verified public endpoint results:

- `GET http://104.154.36.8:8000/health` returned HTTP 200 and `status: ready`.
- `POST http://104.154.36.8:8000/predict` returned HTTP 200 with prediction `0` and confidence `0.814063`.
- `GET http://104.154.36.8:8000/metrics` returned HTTP 200 and Prometheus request/latency metrics.

The final video link should be recorded in `video-link.md` after uploading the student's own screen recording. Do not commit credentials, service-account keys, access tokens, or patient data.
