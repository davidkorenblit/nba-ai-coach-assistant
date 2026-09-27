# 🛡️ Git Workflow Rules & Repository Governance

## 1. Branch Strategy
- **NEVER** push experimental or active development directly to `master` / `main`.
- All scientific development, bug fixes, refactoring, and experiments for the academic paper must happen on the dedicated branch:
  ```bash
  git checkout -b paper-preparation
  # or
  git checkout paper-preparation
  ```
- Merges to `master` require clean validation and explicit review.

---

## 2. Staging & Commits Policy
- **DO NOT** use `git add .` or `git add -A` blindly.
- Always stage specific files explicitly by name:
  ```bash
  git add scripts/feature_engineering/02_build_level2_momentum.py
  git add docs/reports/codebase_audit_and_academic_roadmap.md
  ```
- Review staged changes before committing:
  ```bash
  git status
  git diff --staged
  ```

---

## 3. Data & Artifact Quarantine (Zero Data in Git)
- **STRICTLY FORBIDDEN:** Committing large data files (`.csv`, `.parquet`, `.zip`, `.pkl`, `.joblib`).
- The `data/` directory (pureData, interim, processed, lookup) is purely local and tracked via `.gitignore`.
- Secrets, credentials, and `.env` files must NEVER be staged or committed.

---

## 4. Standard Commit Message Convention
Use semantic prefixes for clear auditability:
- `feat:` New feature engineering / algorithm logic
- `fix:` Bug fix in existing pipeline or model
- `refactor:` Code cleanup, optimization, or modularization without changing behavior
- `docs:` Documentation, reports, decisions, or research notes
- `test:` Validation scripts and test suites
