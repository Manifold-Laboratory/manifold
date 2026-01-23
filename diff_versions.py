import filecmp
import os
import difflib

dir1 = r"D:\ASAS\manifold\gfn"
dir2 = r"D:\ASAS\GFN_test\manifold\gfn"

dcmp = filecmp.dircmp(dir1, dir2)

def print_diff(dcmp):
    for name in dcmp.diff_files:
        print(f"Diff in {name} found in {dcmp.left} and {dcmp.right}")
        with open(os.path.join(dcmp.left, name), 'r') as f1, open(os.path.join(dcmp.right, name), 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            for line in difflib.unified_diff(lines1, lines2, fromfile=os.path.join(dcmp.left, name), tofile=os.path.join(dcmp.right, name)):
                print(line, end='')
                
    for sub_dcmp in dcmp.subdirs.values():
        print_diff(sub_dcmp)

print_diff(dcmp)
print("Finished diff.")
