import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MovieLens Dashboard", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("movie_ratings.csv")
    df["title"] = df["title"].str.strip()
    df["genre_list"] = df["genres"].fillna("").str.split("|").apply(
        lambda gs: [g for g in gs if g and g != "unknown"]
    )
    return df


df = load_data()

st.title("MovieLens Ratings Dashboard")
st.markdown(
    f"{len(df):,} ratings of {df['movie_id'].nunique():,} movies "
    f"by {df['user_id'].nunique():,} users."
)

# --- Q1 ---------------------------------------------------------------------
st.header("1. Genre Breakdown")
st.markdown(
    """
**How multi-genre movies are handled:** a movie can be tagged with several genres
(e.g. *L.A. Confidential* is `Crime|Film-Noir|Mystery|Thriller`). Rather than picking
one "primary" genre, each movie is counted **once in every genre it belongs to**.
Each movie is counted only once per genre no matter how many ratings it received.
Because of this, the genre counts add up to more than the number of movies, so the
percentages are "share of movies tagged with this genre" and do not sum to 100%.
Movies whose only genre is `unknown` are excluded.
"""
)

movies = df.drop_duplicates("movie_id")[["movie_id", "genre_list"]]
movies = movies[movies["genre_list"].str.len() > 0]
genre_counts = (
    movies.explode("genre_list")["genre_list"]
    .value_counts()
    .rename_axis("genre")
    .reset_index(name="movies")
)
genre_counts["share"] = genre_counts["movies"] / len(movies)

c1, c2, c3 = st.columns(3)
c1.metric("Movies counted", f"{len(movies):,}")
c2.metric("Genre tags", f"{genre_counts['movies'].sum():,}")
c3.metric("Avg genres per movie", f"{movies['genre_list'].str.len().mean():.2f}")

q1 = alt.Chart(genre_counts).mark_bar().encode(
    x=alt.X("movies:Q", title="Number of movies tagged with genre"),
    y=alt.Y("genre:N", sort="-x", title=None),
    tooltip=[
        alt.Tooltip("genre:N", title="Genre"),
        alt.Tooltip("movies:Q", title="Movies"),
        alt.Tooltip("share:Q", title="Share of movies", format=".1%"),
    ],
)
q1_labels = q1.mark_text(align="left", dx=3).encode(
    text=alt.Text("share:Q", format=".1%")
)
st.altair_chart((q1 + q1_labels).properties(height=500), width="stretch")

# --- Q2 ---------------------------------------------------------------------
st.header("2. Genre Satisfaction")
st.markdown(
    "Each rating counts toward every genre of the movie it was given to. "
    "The dashed line marks the overall mean rating."
)

rated = df[df["genre_list"].str.len() > 0].explode("genre_list")
genre_ratings = (
    rated.groupby("genre_list")["rating"]
    .agg(mean="mean", count="count")
    .rename_axis("genre")
    .reset_index()
    .sort_values("mean", ascending=False)
)
overall_mean = df["rating"].mean()

top3 = genre_ratings.head(3)
bottom3 = genre_ratings.tail(3).iloc[::-1]
fmt = lambda t: ", ".join(f"**{g}** ({m:.2f})" for g, m in zip(t["genre"], t["mean"]))
st.markdown(f"Highest rated: {fmt(top3)}  \nLowest rated: {fmt(bottom3)}")

q2 = alt.Chart(genre_ratings).mark_bar().encode(
    x=alt.X("mean:Q", title="Mean rating", scale=alt.Scale(domain=[0, 5])),
    y=alt.Y("genre:N", sort="-x", title=None),
    color=alt.Color(
        "mean:Q",
        scale=alt.Scale(scheme="redblue", domainMid=overall_mean),
        legend=None,
    ),
    tooltip=[
        alt.Tooltip("genre:N", title="Genre"),
        alt.Tooltip("mean:Q", title="Mean rating", format=".2f"),
        alt.Tooltip("count:Q", title="Ratings", format=","),
    ],
)
q2_labels = q2.mark_text(align="left", dx=3, color="gray").encode(
    text=alt.Text("mean:Q", format=".2f"), color=alt.value("gray")
)
q2_rule = alt.Chart(pd.DataFrame({"mean": [overall_mean]})).mark_rule(
    strokeDash=[4, 4]
).encode(x="mean:Q")
st.altair_chart(
    (q2 + q2_labels + q2_rule).properties(height=500), width="stretch"
)

# --- Q3 ---------------------------------------------------------------------
st.header("3. Ratings Over Time")
st.markdown(
    "Mean rating grouped by the movie's **release year**. Early years have few "
    "ratings, so their means are noisy; hover a point to see its rating count."
)

by_year = (
    df.dropna(subset=["year"])
    .assign(year=lambda d: d["year"].astype(int))
    .groupby("year")["rating"]
    .agg(mean="mean", count="count", total="sum")
    .reset_index()
)

q3 = alt.Chart(by_year).mark_line(point=True).encode(
    x=alt.X("year:Q", title="Release year", axis=alt.Axis(format="d")),
    y=alt.Y("mean:Q", title="Mean rating", scale=alt.Scale(zero=False)),
    tooltip=[
        alt.Tooltip("year:Q", title="Year", format="d"),
        alt.Tooltip("mean:Q", title="Mean rating", format=".2f"),
        alt.Tooltip("count:Q", title="Ratings", format=","),
    ],
)
layers = q3
if st.checkbox("Show 5-year rolling mean (weighted by rating count)"):
    roll = by_year[["total", "count"]].rolling(5, center=True, min_periods=1).sum()
    by_year["rolling"] = roll["total"] / roll["count"]
    layers = q3 + alt.Chart(by_year).mark_line(color="orange", strokeWidth=3).encode(
        x="year:Q", y="rolling:Q"
    )
st.altair_chart(layers.properties(height=400), width="stretch")

# --- Q4 ---------------------------------------------------------------------
st.header("4. Best Movies, With a Floor")
st.markdown(
    "Top 5 movies by mean rating, counting only movies with at least the chosen "
    "number of ratings. Ties are broken by number of ratings."
)

movie_stats = (
    df.groupby(["movie_id", "title"])["rating"]
    .agg(mean="mean", count="count")
    .reset_index()
)


def top_movies(floor, n=5):
    eligible = movie_stats[movie_stats["count"] >= floor]
    return eligible.sort_values(["mean", "count"], ascending=False).head(n), len(eligible)


floor = st.slider("Minimum number of ratings", 10, 500, 50, step=10)
top, n_eligible = top_movies(floor)
st.caption(f"{n_eligible:,} movies have at least {floor} ratings.")
if n_eligible < 5:
    st.info(f"Only {n_eligible} movie(s) meet this floor.")

q4 = alt.Chart(top).mark_bar().encode(
    x=alt.X("mean:Q", title="Mean rating", scale=alt.Scale(domain=[0, 5])),
    y=alt.Y("title:N", sort="-x", title=None, axis=alt.Axis(labelLimit=400)),
    tooltip=[
        alt.Tooltip("title:N", title="Movie"),
        alt.Tooltip("mean:Q", title="Mean rating", format=".3f"),
        alt.Tooltip("count:Q", title="Ratings", format=","),
    ],
)
q4_labels = q4.mark_text(align="left", dx=3).encode(
    text=alt.Text("mean:Q", format=".2f")
)
st.altair_chart((q4 + q4_labels).properties(height=300), width="stretch")

st.subheader("What changes from a floor of 50 to 150?")
top50, n50 = top_movies(50)
top150, n150 = top_movies(150)
dropped = [t for t in top50["title"] if t not in set(top150["title"])]
entered = [t for t in top150["title"] if t not in set(top50["title"])]
lines = [f"Eligible movies shrink from **{n50:,}** to **{n150:,}**."]
if dropped:
    lines.append("Drop out of the top 5: " + ", ".join(f"*{t}*" for t in dropped))
if entered:
    lines.append("Enter the top 5: " + ", ".join(f"*{t}*" for t in entered))
if not dropped and not entered:
    lines.append("The top 5 is unchanged.")
st.markdown("  \n".join(lines))
