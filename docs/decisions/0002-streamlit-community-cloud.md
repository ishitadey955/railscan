# 0002 Hosting on Streamlit Community Cloud

**Context.** Free hosting for a student/demo project, deploy straight from GitHub.

**Decision.** Streamlit Community Cloud, app redeployed from `main` on every push. Accounts live outside it
(decision 0006), so the host can be restarted or replaced without data loss.

**Alternatives.** Render / Railway / a VM: more control and resources, but paid or more setup.

**Consequences.** Limited, shared CPU and memory: the app can be throttled after sustained CPU use
(seen on 30 Oct 2026) and sleeps when unused. Hence decisions 0003 and 0004, rate-limited scans, and loading
the scanner models once per server start. Every push restarts the app: reboot after a push (runbooks/deploy.md).
