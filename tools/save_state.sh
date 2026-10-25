#!/usr/bin/env bash
# Commit dan push isi state/ ke branch ini, dengan retry + rebase saat run lain
# menulis duluan. Disalin dari RNT (pola Crypto-MEX).
#
#   bash tools/save_state.sh "pesan commit"
#
# Konflik saat rebase:
#   state/*.csv  -> digabung otomatis (merge=union di .gitattributes)
#   state/*.json -> versi run INI yang menang (hanya watcher yang menulis JSON
#                   strategi; watchdog menulis file sendiri)
#   control/bot.yaml -> versi run INI yang menang. Watcher hanya menulisnya saat
#                   mode off otomatis (feed funding putus 14 hari); selain itu file
#                   ini sama dengan origin dan tidak ikut ter-commit.
set -uo pipefail

MSG="${1:-state: $(date -u +%Y-%m-%dT%H:%MZ)}"
BRANCH="${GITHUB_REF_NAME:-main}"

git config user.name  "fcl-bot"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

git add state/ control/bot.yaml
if git diff --cached --quiet; then
  # Tidak ada perubahan baru, tapi mungkin ada commit lama yang gagal di-push.
  git fetch -q origin "${BRANCH}" || true
  if [ -z "$(git rev-list "origin/${BRANCH}..HEAD" 2>/dev/null)" ]; then
    echo "[save_state] tidak ada perubahan state"
    exit 0
  fi
  echo "[save_state] ada commit yang belum ter-push dari percobaan sebelumnya"
else
  git commit -q -m "${MSG} [skip ci]"
fi

for i in 1 2 3 4 5; do
  if git push -q origin "HEAD:${BRANCH}"; then
    echo "[save_state] state tersimpan"
    exit 0
  fi
  echo "[save_state] push ditolak; menggabungkan dengan origin (percobaan $i)"
  git fetch -q origin "${BRANCH}"
  if ! git rebase "origin/${BRANCH}"; then
    for f in $(git diff --name-only --diff-filter=U -- 'state/*.json' control/bot.yaml 2>/dev/null); do
      git checkout --theirs -- "$f" && git add "$f"
    done
    git add state/ control/bot.yaml
    if git grep -qI --cached '^<<<<<<< ' -- state/ 2>/dev/null; then
      echo "::error::penanda konflik tersisa di state/, dibatalkan"
      git rebase --abort || true
      exit 1
    fi
    # JANGAN "rebase --skip": itu membuang commit state run ini diam-diam.
    if ! GIT_EDITOR=true git rebase --continue; then
      git rebase --abort || true
    fi
  fi
  sleep $((i * 3))
done

echo "::error::gagal menyimpan state setelah 5 percobaan"
exit 1
