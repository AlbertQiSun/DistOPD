# Sourced fragment -- not executable on its own. Its variables come from its caller, which has
# already sourced a site file.
#
# ONE COPY OF THE CROSS-SITE CHECK. It has two callers with different appetites:
#
#   slurm/env.sh       requires the ASSETS too (a job about to load pi_ref should not discover
#                      halfway in that the checkpoint is not on this filesystem)
#   slurm/setup_env.sh requires only the ROOT. Building a site's conda env BEFORE staging its
#                      assets is legitimate and, on a fresh cluster, the normal order.
#
# It exists because the check was in env.sh alone, and setup_env.sh walked straight past a
# missing root into `mkdir -p /path/to/scratch/cache/tmp` -- which fails as "Permission
# denied" at the filesystem root, a message that says nothing about the actual mistake.

opd_require_student_ckpt() {               # opd_require_student_ckpt
  # THE STUDENT CHECKPOINT IS A PER-JOB INPUT, NOT A SITE ASSET, and that is why it is no
  # longer in opd_require_paths. Three pi_ref checkpoints now exist -- one per teacher line --
  # and the site cannot know which one a submission meant. It can only supply a default, and a
  # default here is how a job silently rolls out the previous round's checkpoint: they are all
  # the same 4B student, so the wrong one loads cleanly and reports nothing.
  #
  # It moved OUT of the site guard because that guard runs in EVERY job, including the
  # retriever and the teacher server, which never load a student. Requiring it there forces
  # those jobs to name a checkpoint they do not use, and the pressure that creates is to name
  # whichever path happens to exist.
  #
  # So: jobs that load the student call this; jobs that do not, do not.
  if [ -z "${OPD_STUDENT_CKPT:-}" ] || [ ! -d "$OPD_STUDENT_CKPT" ]; then
    echo "FATAL: this job loads the student, and OPD_STUDENT_CKPT is not a directory:" >&2
    echo "         '${OPD_STUDENT_CKPT:-<unset>}'" >&2
    echo "       Three pi_ref checkpoints exist, one per teacher line. Name the one this run" >&2
    echo "       means, at SUBMIT time -- the site default is not one of them:" >&2
    echo '         OPD_STUDENT_CKPT=$OPD_MODELS_DIR/pi_ref_miro bash slurm/submit.sh ...' >&2
    return 1
  fi
  echo "[guard] student checkpoint OK: $OPD_STUDENT_CKPT"
}

opd_require_paths() {                      # opd_require_paths <path>...
  local p
  for p in "$@"; do
    [ -e "$p" ] && continue
    echo "FATAL: OPD_SITE=${OPD_SITE_NAME:-?} declares a path that does not exist here:" >&2
    echo "         $p" >&2
    echo "       Either the wrong site is exported for this cluster ($(hostname -s)), or this" >&2
    echo "       site's tree was never created. Refusing before anything is written." >&2
    return 1
  done
}
