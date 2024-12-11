# cs229_soccer

The document for this project is on: [Soccer Pose Processing](https://office365stanford-my.sharepoint.com/:w:/r/personal/nolanj_stanford_edu/_layouts/15/Doc.aspx?sourcedoc=%7B0DDF1E2A-05E3-4506-83C8-CD8DAC56952F%7D&file=Soccer%20Pose%20Processing.docx&action=default&mobileredirect=true&DefaultItemOpen=1&ct=1730601042322&wdOrigin=OFFICECOM-WEB.MAIN.REC&cid=422d5b74-ac74-4876-ab88-b936c9ddf90d&wdPreviousSessionSrc=HarmonyWeb&wdPreviousSession=08ec8829-6d5f-44b8-961b-a731dcd0934e) and [Brainstorming Space](https://office365stanford-my.sharepoint.com/:w:/r/personal/nolanj_stanford_edu/_layouts/15/doc2.aspx?sourcedoc=%7BA93DE902-FAFF-44E5-8EE9-0F2A87200122%7D&file=Machine%20Learning%20projects%20-%20brainstorm%20space.docx&action=default&mobileredirect=true&DefaultItemOpen=1&nav=eyJjIjozOTM0MzYxNDJ9&ct=1730601031923&wdOrigin=OFFICECOM-WEB.MAIN.REC&cid=53b5681b-28ba-4137-879d-a5070c274ddd&wdPreviousSessionSrc=HarmonyWeb&wdPreviousSession=08ec8829-6d5f-44b8-961b-a731dcd0934e).

Relevant paper: [Applying-Pose-Estimation-to-Predict-Amateur-Golf-Swing-Performance-using-Edge-Processing-February-2020](https://www.researchgate.net/publication/343446840_Applying_Pose_Estimation_to_Predict_Amateur_Golf_Swing_Performance_using_Edge_Processing_February_2020/fulltext/5f2aa704458515b72903a0fe/Applying-Pose-Estimation-to-Predict-Amateur-Golf-Swing-Performance-using-Edge-Processing-February-2020.pdf)

To ensure that YOLOv5 can detect the ball, the ".png" of contact_frames are actually the frames before the soccer is kicked.

## Dataset
- Data_labels.npy:
Each row represents a training example. The columns represent different labels. From 0 -> 6 they are as follows:
0. kick number,
1. foot (left or right), 
2. direction (left=0, center=1, or right=2), 
3. height (low=0, center=1, high=2), 
4. type (curl=0, laces=1), 
5. quality (poor=1, average=2, good=3), 
6. spin (right, left, top, back, knuckle). 

## Preprocessing Result
See file \output\Preprocessed_keypoints_1.npy. This is the numpy array containing the preprocessed data. It is of shape (20,2,25,2). (num_samples, frame, keypoints, x/y). Where frame[0] = contact frame, and frame[1] = plant foot frame. Plant foot frame refers to the frame when the foot-not-hitting-the-ball’s location is fixed.