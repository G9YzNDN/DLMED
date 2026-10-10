# Topic 2 — slide copy and Thai speaking notes

Use the team's shared deck template. These are content drafts for slides **9 and 12**, following [TASKS.md](../TASKS.md); the final deck is assembled by the group. Keep the wording consistent with the generated [results](RESULTS.md).

## Slide 9 — Classification method (~50 seconds)

**Title:** Classifying three tumor types from shared MRI features

**On-slide content (English):**

- Classes: meningioma · glioma · pituitary tumor
- Shared ResNet-34 features → average pooling → dropout 0.2 → linear layer → 3 logits
- Training: cross-entropy loss; evaluation: softmax → argmax
- Metrics: macro Precision / Recall / F1 + confusion matrix

**Visual:** a single flow diagram using the above sequence. Emphasise that the encoder is shared with segmentation. Refer to [the architecture figure](../results/figures/0_architecture.png) if needed.

**Speaking notes (Thai):**

“ส่วนที่ผมรับผิดชอบคือการจำแนกชนิดเนื้องอกสามประเภท ได้แก่ meningioma, glioma และ pituitary tumor ครับ โมเดลรับภาพ MRI หนึ่งภาพ แล้วใช้ ResNet-34 encoder ตัวเดียวกับงาน segmentation เพื่อดึงคุณลักษณะของภาพ จากนั้น classification head จะเฉลี่ยคุณลักษณะในแต่ละช่อง ใช้ dropout และ linear layer เพื่อให้คะแนนสามคลาส ตอนฝึกใช้ cross-entropy ส่วนตอนประเมินใช้ softmax และเลือกคลาสที่มีความน่าจะเป็นสูงที่สุด เราวัดผลด้วย Precision, Recall และ Macro-F1 เพราะแต่ละคลาสมีจำนวนภาพไม่เท่ากัน ทั้งนี้ dataset ไม่มีภาพสมองปกติ จึงเป็นงานจำแนกชนิดในภาพที่มีเนื้องอกครับ”

## Slide 12 — Classification results (~70 seconds)

**Title:** Joint learning improved classification on this test split

**On-slide table (test Macro-F1, mean ± sample SD, 3 seeds):**

| A: Classification only | C: Multi-task equal | D: Multi-task uncertainty |
|---|---|---|
| 0.916 ± 0.010 | **0.938 ± 0.003** | 0.923 ± 0.012 |

**Two takeaways:**

- C−A: **+0.022 Macro-F1**, positive in all 3 seeds.
- Meningioma is the weakest class; a single patient group accounts for 54 of C's 63 errors pooled over seeds.

**Visual:** [confusion matrices](../results/figures/3_confusion_matrices.png). Use A blue, C green and D yellow for experiment labels, matching the team palette. If a full-size matrix figure crowds the table, use the table in slide 11 and show the matrix with the +0.022 callout on slide 12.

**Caption:** Test: 436 slices / 29 patient groups. Matrices pool 3 seeds: 1,308 predictions on the same slices. Rows = true; columns = predicted. Error counts are repeated predictions, not independent patients.

**Speaking notes (Thai):**

“ผลบน test set แสดงว่าโมเดลที่ฝึก classification อย่างเดียวมี Macro-F1 เฉลี่ยประมาณ 0.916 ส่วน multi-task ที่ให้น้ำหนักสองงานเท่ากันได้ 0.938 เพิ่มขึ้นประมาณ 0.022 และดีกว่าทั้งสาม seeds ครับ แบบเรียนรู้น้ำหนักอัตโนมัติได้ประมาณ 0.923 สำหรับ confusion matrix แถวคือคลาสจริง และคอลัมน์คือคลาสที่ทำนาย จุดบนแนวทแยงคือคำตอบที่ถูก คลาสที่ยากที่สุดคือ meningioma โดยผู้ป่วยหนึ่งกลุ่มถูกทายเป็น pituitary ทั้ง 18 ภาพในทุก seed รวม 54 จาก 63 ข้อผิดพลาดของโมเดล C การกำกับบริเวณเนื้องอกอาจช่วย encoder เรียนคุณลักษณะที่จำแนกได้ดีขึ้น แต่เรายังไม่ได้พิสูจน์กลไกนี้ และผลจากสาม seeds กับผู้ป่วย test 29 กลุ่มยังไม่ใช่ข้อสรุปเรื่องนัยสำคัญทางสถิติครับ”

## Backup material

- [Per-class F1 table](RESULTS.md#per-class-f1) for questions about the hardest class.
- [Misclassified examples](../results/figures/7_C_mtl_equal_seed0_misclassified.png) for the error-analysis presenter to discuss.
- [QA.md](QA.md) for short answers and handoffs to teammates.

## Rehearsal checks

- Time both sections together; the timings above are targets, not measured speech durations.
- Say “+0.022 Macro-F1,” not “accuracy improved by 2.2%.”
- Say “lowest F1 in this test split,” not “meningioma is always hardest.”
- Describe location/appearance explanations as hypotheses from the repository analysis, not confirmed radiological findings.
- Explain that the original runs preceded the augmentation-seeding fix; same seed labels did not control augmentation draws.
- Do not remove the difficult patient from the main comparison or describe pooled seeds as independent test subjects.
