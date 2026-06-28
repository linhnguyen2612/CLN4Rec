# CL4Rec — Contrastive Learning for Recommendation (ACIIDS 2024)

Mã nguồn cho mô hình gợi ý dựa trên **Knowledge Graph + Contrastive Learning**
(kèm các baseline VAE / KGAT / BPRMF / NFM …), thực nghiệm trên bộ
**MovieLens 100K**.

## Cấu trúc thư mục

```
CL4Rec/
├── main_*.py                 # Các script huấn luyện chính (entry points)
│   ├── main_graph_bi_interaction_with_ufo_space.py
│   ├── main_graph_bi_interaction_multi_domain.py
│   ├── main_ae_with_ufo_space_multi_domain.py
│   └── main_ae_raw_multi_domain.py
├── mask_optimization_*.py    # Tối ưu mask cho UFO-space / VAE
├── model/                    # Định nghĩa kiến trúc mô hình
│   ├── Graph_Bi_interaction_with_UFO_SPACE.py
│   ├── VAE.py
│   └── VAE_raw.py
├── data_loader/              # Nạp & tiền xử lý dữ liệu
│   ├── loader_base.py
│   ├── loader_kgat.py
│   └── loader_VAE.py
├── parsers/                  # Tham số dòng lệnh cho từng mô hình
│   ├── parser_kgat.py, parser_vae.py, parser_bprmf.py, ...
├── utils/                    # Tiện ích (metrics, log, t-SNE, ...)
│   ├── metrics.py
│   ├── log_helper.py
│   ├── model_helper.py
│   └── tSNE_visualize.py
├── datasets/                 # Dữ liệu đã tiền xử lý (train/test/val + KG)
│   ├── ml_100k/
│   └── ml_100k_action_fantasy/
├── ml-100k/                  # Dữ liệu thô MovieLens 100K (~15 MB)
├── ml_100k_eda.ipynb         # Notebook tiền xử lý / EDA
├── run_ufo_space.ipynb       # Notebook chạy thực nghiệm & so sánh
├── paper_193.pdf             # Bài báo
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
git clone <repo-url>
cd CL4Rec
python -m venv .venv && source .venv/bin/activate   # tùy chọn
pip install -r requirements.txt
```

## Cách chạy

Tiền xử lý dữ liệu MovieLens 100K:

```bash
jupyter nbconvert --to notebook --execute ml_100k_eda.ipynb
```

Huấn luyện mô hình đề xuất + baseline (so sánh hiệu năng & ablation study):

```bash
jupyter nbconvert --to notebook --execute run_ufo_space.ipynb
```

## Dữ liệu MovieLens 100K

Thư mục `ml-100k/` là dữ liệu thô tải công khai từ
[GroupLens](https://grouplens.org/datasets/movielens/100k/).
Nếu bạn loại nó khỏi git (bỏ comment `ml-100k/` trong `.gitignore`),
người dùng có thể tải lại:

```bash
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```
