# Roll back

The deployed version is whatever is on `main`. To go back to the previous version:

```bat
git log --oneline -5                 :: find the last good commit, e.g. 99ec7e6
git revert --no-edit HEAD            :: undo the last commit with a new commit (history is kept)
git push
```
Then **Manage app -> Reboot app** and run the checks in deploy.md step 4.

Undo several commits: `git revert --no-edit <good-commit>..HEAD`.

Database: the accounts schema only ever grows (new tables/columns), so an older app version works with a
newer database. Restore data only for data loss, never for a code rollback (backups.md).
