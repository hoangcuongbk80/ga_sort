# GlareMOT sample sequences

Two real sequences from the GlareMOT test set, included to show the exact on-disk
structure of the released dataset (matches Section 3 of the paper).

- `GlareMOT/204_sequence_81q/`
- `GlareMOT/145_sequence_28q/`

Each sequence directory contains:
- `images/` - **subsampled** here (~25 evenly-spaced frames per sequence) to keep
                    the supplementary archive under the size limit. The full sequences
                    (542 / 1386 frames respectively) ship with the public dataset release.
- `labels/` - visible-only YOLO-format person boxes (5-col: cls cx cy w h), COMPLETE
                    (all frames, not subsampled).
- `labels_all/` - amodal YOLO-format boxes, i.e. `labels/` plus boxes kept through glare/
                    occlusion (Section 3.2, "Annotation Protocol"). COMPLETE, all frames.
- `mot/gt/gt.txt` - MOTChallenge 9-column tracking ground truth (frame,id,x,y,w,h,conf,
                    class,vis), amodal identity kept continuous through burn. COMPLETE.
- `meta/` - per-sequence build metadata (frame/video timeline alignment,
                    uncertain-frame log). COMPLETE.

Note: because `images/` is subsampled, `labels/`, `labels_all/`, and `mot/gt/gt.txt` contain
rows for frames that have no corresponding image file in this sample; that is expected and
lets reviewers inspect the full annotation density against a representative visual sample.
