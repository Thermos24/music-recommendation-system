import re
import warnings
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# ================================================================
# ĐỌC DỮ LIỆU
# ================================================================

data = pd.read_csv("tcc_ceds_music.csv")
df = data.copy()

num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

LABEL_COL = "genre"

SCALE_COLS = [
    c for c in num_cols
    if c not in ["Unnamed: 0", "release_date", "len", "age"]
]


# ================================================================
# TIỀN XỬ LÝ DỮ LIỆU
# ================================================================

for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

df.dropna(inplace=True)

data["track_name"] = data["track_name"].fillna("").astype(str).str.strip()
data["artist_name"] = data["artist_name"].fillna("").astype(str).str.strip()
data["genre"] = data["genre"].fillna("").astype(str).str.strip()
data["topic"] = data["topic"].fillna("").astype(str).str.strip()
data["lyrics"] = data["lyrics"].fillna("").astype(str).str.strip()

data["_track_lower"] = data["track_name"].astype(str).str.lower().str.strip()
data["_artist_lower"] = data["artist_name"].astype(str).str.lower().str.strip()
data["_lyrics_lower"] = data["lyrics"].astype(str).str.lower().str.strip()


# ================================================================
# VISUALIZATION
# ================================================================

def plot_eda_three(data: pd.DataFrame):
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle(
        "Khám phá dữ liệu: Thể loại và Chủ đề",
        fontsize=15,
        fontweight="bold"
    )

    genre_order = data["genre"].value_counts().index

    sns.countplot(
        y="genre",
        data=data,
        order=genre_order,
        palette="Blues_d",
        ax=axes1[0]
    )

    axes1[0].set_title("Số bài theo thể loại", fontweight="bold")
    axes1[0].set_xlabel("Số bài")
    axes1[0].set_ylabel("Thể loại")

    for p in axes1[0].patches:
        axes1[0].text(
            p.get_width() + 30,
            p.get_y() + p.get_height() / 2,
            f"{int(p.get_width()):,}",
            va="center",
            fontsize=9
        )

    topic_order = data["topic"].value_counts().index

    sns.countplot(
        y="topic",
        data=data,
        order=topic_order,
        palette="Greens_d",
        ax=axes1[1]
    )

    axes1[1].set_title("Số bài theo chủ đề", fontweight="bold")
    axes1[1].set_xlabel("Số bài")
    axes1[1].set_ylabel("Chủ đề")

    for p in axes1[1].patches:
        axes1[1].text(
            p.get_width() + 30,
            p.get_y() + p.get_height() / 2,
            f"{int(p.get_width()):,}",
            va="center",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    fig2.suptitle(
        "Khám phá dữ liệu: Nghệ sĩ",
        fontsize=15,
        fontweight="bold"
    )

    top_artists = data.groupby("artist_name").size().sort_values(
        ascending=False
    ).head(30)

    sns.barplot(
        x=top_artists.values,
        y=top_artists.index,
        hue=top_artists.index,
        palette="viridis",
        legend=False,
        ax=ax2
    )

    ax2.set_title("Top 30 nghệ sĩ theo số bài", fontweight="bold")
    ax2.set_xlabel("Số bài")
    ax2.set_ylabel("Nghệ sĩ")

    for p in ax2.patches:
        if p.get_width() > 0:
            ax2.text(
                p.get_width() + 0.5,
                p.get_y() + p.get_height() / 2,
                str(int(p.get_width())),
                va="center",
                fontsize=8
            )

    plt.tight_layout()
    plt.show()


def plot_eda_section():
    corr_matrix = df[SCALE_COLS].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(16, 12))

    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        mask=mask,
        linewidths=0.4,
        square=True,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.75}
    )

    plt.title(
        "Correlation Heatmap — Numerical Features",
        fontweight="bold",
        fontsize=15,
        pad=20
    )

    plt.xticks(rotation=60, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show()

    feature_pages = [
        SCALE_COLS[:9],
        SCALE_COLS[9:18],
        SCALE_COLS[18:]
    ]

    for page_cols in feature_pages:
        if len(page_cols) == 0:
            continue

        if len(page_cols) <= 4:
            nrows, ncols = 2, 2
            figsize = (12, 8)
        else:
            nrows, ncols = 3, 3
            figsize = (15, 11)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle(
            "Histogram của các Feature",
            fontsize=13,
            fontweight="bold"
        )

        axes = axes.ravel()

        for i, col in enumerate(page_cols):
            axes[i].hist(
                df[col].dropna(),
                bins=30,
                color="steelblue",
                edgecolor="k",
                alpha=0.85
            )

            axes[i].set_title(col, fontsize=10, fontweight="bold")
            axes[i].set_xlabel("Giá trị")
            axes[i].set_ylabel("Tần suất")
            axes[i].grid(alpha=0.25)

        for j in range(len(page_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    for page_cols in feature_pages:
        if len(page_cols) == 0:
            continue

        if len(page_cols) <= 4:
            nrows, ncols = 2, 2
            figsize = (12, 8)
        else:
            nrows, ncols = 3, 3
            figsize = (15, 11)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle(
            "Boxplot — Phát hiện Outlier",
            fontsize=13,
            fontweight="bold"
        )

        axes = axes.ravel()

        for i, col in enumerate(page_cols):
            sns.boxplot(
                y=df[col],
                ax=axes[i],
                color="lightcoral"
            )

            axes[i].set_title(col, fontsize=10, fontweight="bold")
            axes[i].set_ylabel("Giá trị")
            axes[i].grid(alpha=0.25)

        for j in range(len(page_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ================================================================
# CORE LOGIC — TF-IDF + COSINE SIMILARITY
# ================================================================

data["combined_features"] = (
    data["genre"].fillna("") + " " +
    data["artist_name"].fillna("") + " " +
    data["track_name"].fillna("")
)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["combined_features"])


def get_recommendations(song_title, data, tfidf_matrix, top_n=10):
    idx_list = data[data["track_name"] == song_title].index

    if len(idx_list) == 0:
        print("Song not found in the dataset.")
        return pd.DataFrame()

    base_idx = idx_list[0]

    song_vector = tfidf_matrix[base_idx]
    scores = cosine_similarity(song_vector, tfidf_matrix)[0]

    similar_indices = np.argsort(scores)[::-1]
    similar_indices = [i for i in similar_indices if i != base_idx]
    top_indices = similar_indices[:top_n]

    recommendations = data.iloc[top_indices].copy()
    recommendations["similarity"] = scores[top_indices]

    return recommendations


# ================================================================
# TÌM KIẾM THÔNG MINH
# ================================================================

KEYWORD_GENRE_MAP = {
    "pop": "pop",
    "dance": "pop",
    "chart": "pop",
    "hit": "pop",
    "mainstream": "pop",
    "radio": "pop",

    "country": "country",
    "cowboy": "country",
    "farm": "country",
    "western": "country",
    "banjo": "country",
    "nashville": "country",
    "honky": "country",
    "twang": "country",

    "blues": "blues",
    "soul": "blues",
    "gospel": "blues",
    "delta": "blues",
    "hurt": "blues",
    "pain": "blues",

    "rock": "rock",
    "guitar": "rock",
    "band": "rock",
    "electric": "rock",
    "metal": "rock",
    "punk": "rock",

    "jazz": "jazz",
    "swing": "jazz",
    "bebop": "jazz",
    "saxophone": "jazz",
    "trumpet": "jazz",
    "improvise": "jazz",
    "smooth": "jazz",

    "reggae": "reggae",
    "jamaican": "reggae",
    "rasta": "reggae",
    "caribbean": "reggae",
    "island": "reggae",
    "rastafari": "reggae",

    "hip": "hip hop",
    "hop": "hip hop",
    "rap": "hip hop",
    "hiphop": "hip hop",
    "rapper": "hip hop",
    "flow": "hip hop",
    "beat": "hip hop",
    "rhyme": "hip hop",

    "sad": "blues",
    "cry": "blues",
    "love": "pop",
    "night": "jazz",
    "party": "pop",
}


def smart_search(keyword: str, data: pd.DataFrame, top_n: int = 10):
    kw = keyword.lower().strip()

    if not kw:
        return pd.DataFrame(), "not_found", ""

    mask_track = data["_track_lower"].str.contains(kw, na=False, regex=False)

    if mask_track.any():
        matched = data[mask_track].copy()

        matched["_search_score"] = matched["_track_lower"].apply(
            lambda t: 3 if t == kw else (2 if t.startswith(kw) else 1)
        )

        matched = matched.sort_values("_search_score", ascending=False)
        return matched.head(top_n), "track", keyword

    mask_artist = data["_artist_lower"].str.contains(kw, na=False, regex=False)

    if mask_artist.any():
        matched = data[mask_artist].copy()

        artist_counts = data.groupby("_artist_lower").size().to_dict()
        matched["_search_score"] = matched["_artist_lower"].map(
            artist_counts
        ).fillna(0)

        matched = matched.sort_values("_search_score", ascending=False)
        return matched.head(top_n), "artist", keyword

    pattern = r"\b" + re.escape(kw) + r"\b"
    mask_lyrics = data["_lyrics_lower"].str.contains(
        pattern,
        na=False,
        regex=True
    )

    if mask_lyrics.any():
        matched = data[mask_lyrics].copy()

        matched["_search_score"] = matched["_lyrics_lower"].str.count(
            pattern,
            flags=re.IGNORECASE
        )

        matched = matched.sort_values("_search_score", ascending=False)
        return matched.head(top_n), "lyrics", keyword

    words = kw.split()

    for word in words:
        if word in KEYWORD_GENRE_MAP:
            inferred_genre = KEYWORD_GENRE_MAP[word]
            mask_genre = data["genre"].str.lower() == inferred_genre

            if mask_genre.any():
                matched = data[mask_genre].copy()

                return matched.sample(
                    min(top_n, len(matched)),
                    random_state=42
                ), "genre_keyword", inferred_genre

    return pd.DataFrame(), "not_found", ""


def get_tfidf_recommendations_for_results(
    search_df,
    search_mode,
    data,
    tfidf_matrix,
    top_n=10
):
    if search_df.empty:
        return pd.DataFrame(), ""

    base_row = search_df.iloc[0]
    base_title = base_row["track_name"]

    recs = get_recommendations(
        base_title,
        data,
        tfidf_matrix,
        top_n=top_n
    )

    return recs, base_title


# ================================================================
# ĐỒ THỊ KẾT QUẢ GỢI Ý
# ================================================================

RADAR_COLS = [
    "sadness",
    "violence",
    "romantic",
    "energy",
    "danceability",
    "valence",
    "obscene",
    "feelings"
]


def plot_recommendations_charts(
    keyword,
    search_mode,
    matched_value,
    base_title,
    search_df,
    recs_df,
    data
):
    if recs_df.empty or "similarity" not in recs_df.columns:
        return

    mode_labels = {
        "track": f'Bài hát  : "{matched_value}"',
        "artist": f'Nghệ sĩ  : "{matched_value}"',
        "lyrics": f'Lyrics   : "{matched_value}"',
        "genre_keyword": f'Thể loại : "{matched_value}"',
    }

    mode_label = mode_labels.get(search_mode, f'"{keyword}"')

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f'Gợi ý cho — {mode_label}  (gốc: "{base_title}")',
        fontsize=13,
        fontweight="bold"
    )

    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], polar=True)

    labels = [
        f"{str(row['track_name'])[:28]}\n({str(row['artist_name'])[:18]})"
        for _, row in recs_df.iterrows()
    ]

    sim_vals = recs_df["similarity"].values

    cmap = plt.cm.RdYlGn
    denom = max(sim_vals.max() - sim_vals.min(), 1e-9)
    normalized_vals = (sim_vals - sim_vals.min()) / denom
    colors = [cmap(v) for v in normalized_vals]

    bars = ax1.barh(
        range(len(recs_df)),
        sim_vals,
        color=colors,
        edgecolor="white",
        linewidth=0.6
    )

    ax1.set_yticks(range(len(recs_df)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Cosine Similarity Score", fontsize=10)
    ax1.set_title("Top 10 bài gợi ý", fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(sim_vals) * 1.12)

    for bar, val in zip(bars, sim_vals):
        ax1.text(
            val + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=7.5,
            color="#333"
        )

    available_radar = [c for c in RADAR_COLS if c in data.columns]

    if len(available_radar) < 3:
        ax2.set_visible(False)
        plt.tight_layout()
        plt.show()
        return

    angles = np.linspace(
        0,
        2 * np.pi,
        len(available_radar),
        endpoint=False
    ).tolist()

    angles += angles[:1]

    base_mask = data["track_name"] == base_title

    if base_mask.any():
        base_vals = data.loc[base_mask, available_radar].mean().values.tolist()
    else:
        base_vals = [0.0] * len(available_radar)

    base_vals += base_vals[:1]

    rec_vals = recs_df[available_radar].mean().values.tolist()
    rec_vals += rec_vals[:1]

    ax2.plot(
        angles,
        base_vals,
        "o-",
        lw=2,
        label="Bài gốc",
        color="#2196F3"
    )

    ax2.fill(
        angles,
        base_vals,
        alpha=0.15,
        color="#2196F3"
    )

    ax2.plot(
        angles,
        rec_vals,
        "s-",
        lw=2,
        label="TB gợi ý",
        color="#FF5722"
    )

    ax2.fill(
        angles,
        rec_vals,
        alpha=0.15,
        color="#FF5722"
    )

    ax2.set_thetagrids(
        np.degrees(angles[:-1]),
        available_radar,
        fontsize=8
    )

    ax2.set_ylim(0, 1)
    ax2.set_title("So sánh hồ sơ cảm xúc", fontweight="bold", pad=20)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ================================================================
# IN KẾT QUẢ TRÊN CMD
# ================================================================

def print_search_result(search_df, search_mode, matched_value, keyword):
    mode_msg = {
        "track": f'Tìm thấy theo TÊN BÀI HÁT "{matched_value}"',
        "artist": f'Tìm thấy theo TÊN NGHỆ SĨ "{matched_value}"',
        "lyrics": f'Tìm thấy theo LYRICS "{matched_value}"',
        "genre_keyword": f'Phân tích keyword → thể loại "{matched_value}"',
        "not_found": f'Không tìm thấy kết quả cho "{keyword}"',
    }

    print(f"\n  {mode_msg.get(search_mode, '')}")

    if search_mode == "not_found" or search_df.empty:
        print("    Gợi ý: thử tên bài khác, tên nghệ sĩ, hoặc từ khoá cảm xúc")
        print("    Các thể loại có sẵn: pop, country, blues, rock, jazz, reggae, hip hop")
        return

    print(f"\n  {'#':<3} {'Tên bài':<30} {'Nghệ sĩ':<22} {'Thể loại':<10} {'Chủ đề'}")
    print("  " + "─" * 75)

    for i, (_, row) in enumerate(search_df.iterrows(), 1):
        print(
            f"  {i:<3} "
            f"{str(row['track_name'])[:28]:<30} "
            f"{str(row['artist_name'])[:20]:<22} "
            f"{str(row['genre']):<10} "
            f"{str(row.get('topic', ''))}"
        )


def print_recommendations(recs_df, base_title):
    if recs_df is None or recs_df.empty:
        print("  [Gợi ý] Không có kết quả.")
        return

    print(f"\n  Top 10 bài gợi ý từ '{base_title}':\n")

    print(
        f"  {'#':<3} "
        f"{'Tên bài':<30} "
        f"{'Nghệ sĩ':<22} "
        f"{'Thể loại':<10} "
        f"{'Chủ đề':<14} "
        f"{'Similarity'}"
    )

    print("  " + "─" * 90)

    for i, (_, row) in enumerate(recs_df.iterrows(), 1):
        print(
            f"  {i:<3} "
            f"{str(row['track_name'])[:28]:<30} "
            f"{str(row['artist_name'])[:20]:<22} "
            f"{str(row['genre']):<10} "
            f"{str(row.get('topic', '')):<14} "
            f"{row.get('similarity', 0):.4f}"
        )


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "═" * 100)
    print("                                       HỆ THỐNG GỢI Ý NHẠC")
    print("═" * 100)

    plot_eda_three(data)
    plot_eda_section()

    while True:
        print()

        keyword = input("Nhập từ khoá tìm kiếm (nhập 0 để thoát): ").strip()

        if keyword == "0":
            print("Tạm biệt! Hẹn gặp lại.\n")
            break

        if not keyword:
            print("Chưa nhập gì. Hãy thử lại.")
            continue

        search_df, search_mode, matched_value = smart_search(
            keyword,
            data,
            top_n=10
        )

        print_search_result(
            search_df,
            search_mode,
            matched_value,
            keyword
        )

        if search_mode == "not_found":
            again = input("Thử từ khoá khác? (Nhập 0 để thoát): ").strip()

            if again == "0":
                print("Tạm biệt!\n")
                break

            continue

        recs_df, base_title = get_tfidf_recommendations_for_results(
            search_df,
            search_mode,
            data,
            tfidf_matrix,
            top_n=10
        )

        print_recommendations(recs_df, base_title)

        if not recs_df.empty:
            plot_recommendations_charts(
                keyword,
                search_mode,
                matched_value,
                base_title,
                search_df,
                recs_df,
                data
            )

        print("\n" + "─" * 60)


if __name__ == "__main__":
    main()