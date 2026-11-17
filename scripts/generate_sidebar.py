from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "_sidebar.md"
SKIPPED_FILES = {"_sidebar.md", "_navbar.md", "_coverpage.md", "404.md"}


def page_title(path):
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def contains_markdown(directory):
    return any(
        path.is_file()
        and path.suffix.lower() == ".md"
        and path.name not in SKIPPED_FILES
        for path in directory.rglob("*")
        if ".git" not in path.parts and ".github" not in path.parts
    )


def sidebar_lines(directory, depth=0):
    lines = []
    entries = sorted(
        (
            path
            for path in directory.iterdir()
            if not path.name.startswith(".") and path.name != "scripts"
        ),
        key=lambda path: (path.is_file(), path.name.casefold()),
    )

    for path in entries:
        if path.is_dir():
            if not contains_markdown(path):
                continue
            lines.append(f"{'  ' * depth}- {path.name}")
            lines.extend(sidebar_lines(path, depth + 1))
        elif (
            path.suffix.lower() == ".md"
            and path.name not in SKIPPED_FILES
            and not (path.parent == ROOT and path.name.casefold() == "readme.md")
        ):
            relative_path = path.relative_to(ROOT).as_posix()
            lines.append(
                f"{'  ' * depth}- [{page_title(path)}](<{relative_path}>)"
            )

    return lines


def main():
    lines = ["- [首页](readme.md)", "", *sidebar_lines(ROOT)]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
