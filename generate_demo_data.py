"""Generates a SYNTHETIC dataset ONLY to test that the pipeline runs.
Its results are meaningless. Never report metrics from this data."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 370
yn = lambda p: rng.choice(["Yes", "No"], n, p=[p, 1 - p])
df = pd.DataFrame({
    "gender": rng.choice(["Male", "Female"], n),
    "age_group": rng.choice(["18-20", "21-23", "24-30", "31+"], n, p=[.3, .35, .2, .15]),
    "role": rng.choice(["Student", "Teaching staff", "Administrative staff"], n, p=[.7, .15, .15]),
    "year_or_position": rng.choice(["Year 1", "Year 2", "Year 3", "Year 4", "Staff"], n),
    "uses_whatsapp": yn(.95), "uses_facebook": yn(.5), "uses_x": yn(.3),
    "uses_instagram": yn(.6), "uses_tiktok": yn(.55),
    "hours_per_day": np.round(rng.gamma(3, 1.2, n), 1),
    "purpose_of_use": rng.choice(["Social", "News", "Academic", "Entertainment"], n),
    "main_info_source": rng.choice(["Official notices", "Class WhatsApp groups",
                                    "Student pages", "Friends", "Personal accounts"], n),
    "trust_in_source": rng.integers(1, 6, n),
    "misinfo_exposure": rng.choice(["Never", "Rarely", "Sometimes", "Often", "Very often"], n),
    "checks_official_channels": rng.integers(1, 6, n),
    "follows_leaders_online": yn(.5), "engages_with_posts": yn(.4),
})
score = (df.trust_in_source + df.checks_official_channels) / 2 \
    - df.misinfo_exposure.map({"Never": 0, "Rarely": .3, "Sometimes": .7, "Often": 1.2, "Very often": 1.6})
for c in ["perception_trust", "perception_credibility", "perception_competence", "perception_approachability"]:
    df[c] = np.clip(np.round(score + rng.normal(0, .8, n)), 1, 5).astype(int)
df.to_csv("data/survey_responses.csv", index=False)
print("Wrote data/survey_responses.csv (SYNTHETIC) shape", df.shape)
