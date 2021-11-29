# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import skvideo.io

def get_git_rev(path: str, search_parent_directories: bool = True):
    try:
        import git
    except ImportError:
        print(
            "Warning: gitpython not installed."
            " Unable to log git rev."
            " Run `pip install gitpython` if you want git revs to be logged.")
        return None

    try:
        repo = git.Repo(
            path, search_parent_directories=search_parent_directories)
        git_rev = repo.active_branch.commit.name_rev
    except TypeError:
        git_rev = repo.head.object.name_rev

    return git_rev

def save_video(filename, video_frames, fps=60):
    assert fps == int(fps), fps
    skvideo.io.vwrite(filename, video_frames, inputdict={'-r': str(int(fps))})
