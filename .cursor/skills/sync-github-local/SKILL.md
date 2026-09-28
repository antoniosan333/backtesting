---
name: sync-github-local
description: Use when a git push, pull, or branch checkout should leave GitHub and the user's PC on the same commit, or when the user asks to sync, update, or pull the local checkout.
---

# Sync GitHub and the local PC

## Overview

GitHub is the shared copy. A cloud agent can update GitHub. Only a process on the user's PC can update that PC. Never claim the PC is updated after a push from this environment.

## When to Use

- The user asks to sync, pull, or update the local checkout.
- You are about to commit or push.
- You just pushed and the user expects the same files on their computer.

## Cloud agent

1. Commit the intended files and `git push -u origin <branch>`.
2. Confirm `git status -sb` shows the local branch even with `origin/<branch>`.
3. Tell the user the branch name, commit, and that the PC updates only after the sync script or a local agent runs.
4. Do not invent a remote, SSH session, or copy step to their computer. The only git remote is `origin` on GitHub.

## Local agent (running on the user's PC)

1. At the start of git work, `git fetch origin`.
2. If the current branch has no uncommitted changes and origin is a fast-forward, run `scripts/sync-from-github.ps1`.
3. If the working tree is dirty or the branches have diverged, stop and say so. Do not reset, stash, or force-push to make them match.
4. After a commit, `git push -u origin <current-branch>` so GitHub matches the PC.

## Script

From the repo root on the PC:

```powershell
.\scripts\sync-from-github.ps1
.\scripts\sync-from-github.ps1 -Branch cursor/volatility-analysis-plan-d71b
```

The script fetches `origin` and fast-forwards only. It leaves a dirty tree or diverged branch unchanged and exits with an error.
