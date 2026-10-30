# RailScan system design

## Problem
Travellers in India meet station boards and train signs in scripts they may not read (Devanagari,
Bengali, Kannada, Telugu), and have no easy estimate of how late a train will be further down its route.
RailScan reads a photo of a sign and identifies the station and train numbers. It also estimates delays,
and the position the train should have, from recorded journeys and a delay the user enters.

## Users and tenancy
- **One kind of user:** an individual traveller with a personal account. There are no teams, organisations
  or roles.
- **Tenancy:** per user. Each history entry belongs to one user. Every query filters by that user's id, and
  only that user can read, export or delete it.
- **Shared, read-only data:** the railway data and the models are public and identical for everyone.

## Core flows
1. **Sign up / log in.** Username and password, with lockout after 5 failed attempts in 15 minutes and
   sign-ups limited per IP address.
2. **Scan a sign.** Upload a photo, then: CNN and OCR fusion for the script, OCR text, station match,
   train numbers and platform. The result is saved to the user's history, without the photo.
3. **Journey planner.** Train, date and station, plus an optional current delay. Shows the estimated
   delay at the next and later stops and the estimated position (an estimate, not live).
4. **History.** List, inspect, delete entries; download everything as JSON; delete the account.
5. **Data and models.** Dataset figures and measured model accuracy.

## Data
| Data | Sensitivity | Where |
|---|---|---|
| Accounts: username, display name, scrypt hash | Personal | PostgreSQL (Supabase) |
| History: results shown (no photos) | Personal | PostgreSQL (Supabase) |
| Login attempts (30 days), usage counts (1 day) | Security records | PostgreSQL (Supabase) |
| Railway observations: 924,706 rows, 5,068 trains | Public dataset (GPL-3.0) | Committed Parquet, SQLite locally |
| Models: CNN, delay models | Public | Committed files |

## Non-functional requirements
- **Pages:** open in about 2 s or less after the first page view of a server start, with no page waiting
  for a database build. Measured: login to the Scan page 0.7 s, planner estimate 1.8 s.
- **Scans:** about 20 s on the host's CPU for a phone photo. The OCR runs once per script as validated;
  the faster single-reading variant was measured and rejected (79.2% vs 75.6% test accuracy, before
  faint readings counted; script accuracy is now 86.3%).
- **Memory:** about 1.5 GB at peak during a scan, which must fit the host.
- **Privacy:** passwords are only ever stored hashed; logs are redacted; users can export and delete
  their data.
- **Availability:** best effort on a free host. The app may sleep or be throttled. Accounts live outside
  the host, so a restart loses nothing.

## Out of scope
Live train status, payments and plans, organisations and teams, email (verification and password reset),
a public API, mobile apps, coverage beyond northern India and West Bengal.

## Open questions
- Use a managed auth provider for email verification and password reset? (decision 0005)
- Move to paid hosting if usage grows beyond the free tier? (docs/HOSTING.md)
