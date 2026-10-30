# 0005 Own account handling (status: open for review)

**Context.** Accounts need sign-up, log-in and a private history. The playbook recommends a proven auth
provider instead of own password handling.

**Decision (current).** Own accounts in `src/database/users.py`: scrypt (Python standard library) with a
random salt per user, constant-time comparison, generic login errors, lockout after 5 failed attempts in
15 minutes, sign-ups limited per IP, all SQL parameterised; tested on SQLite and PostgreSQL.

**Alternatives.** Supabase Auth (already using Supabase), Auth0, Clerk: email verification, password reset,
social log-in and session revocation for free, but a redesign of sign-up/log-in and an email setup.

**Consequences.** No email verification and no password reset (a forgotten password means a new account).
Revisit before inviting users beyond a small group; this is the riskiest decision in docs/architecture.md.
