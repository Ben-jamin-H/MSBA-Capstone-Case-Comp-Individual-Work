"""Shared build for the PBS Utah group EDA.

Copied unchanged (apart from this note) from the team's group EDA, written with Erin
Bednarik, Samantha Huang, and Kate Klinger, so that my modeling notebook uses the same
cleaning rules, eligible population, and household rule as the group EDA.

Turns the raw release into cleaned tables and a donor-by-fiscal-year panel using the
rules the team agreed on (see the "Team EDA Synthesis" decisions):

1. Copy rows are removed with content-based rules; the SYN-DEF / SYN-UNMATCHED labels
   are only used afterwards to check the rules.
2. Eligible donors gave in FY t and were not major in FY t; lapsed donors (gave in
   FY t-1 or t-2 but not FY t) are a separate segment.
3. Former major donors stay in and are flagged.
4. Legacy gifts are credited to the first listed person, with spouse rows flagged.
5. Snapshots run FY2021 to FY2025 with legacy history behind them (cohorts back to
   FY1999 are available for giving-only checks).
6. Repeat solicitation rows are kept.
15. Major status is per person, except a couple whose combined giving reaches $1,200
    (when neither spouse does alone) counts as one major donor; both spouses are
    labeled major and share a couple_id so the couple can be counted once.
"""

import re
from pathlib import Path

import polars as pl

MAJOR = 1200
FIRST_COHORT = 1999            # legacy records are complete from FY1999 on
FUNDING_CUT_FY = 2026          # first fiscal year without federal funding
TIERS = [1200, 2000, 5000]
TIER_LABELS = ["under $1,200", "$1,200", "$2,000", "$5,000+"]

# columns that need a type other than text, by table
TYPES = {
    "constituents_w_memberships": {
        "Age": "num", "UofU Degree Count": "num", "Latest UofU Degree Year": "num",
        "Lifetime UofU Fundraising": "num", "Previous FY UofU Cash": "num",
        "Prospect Committed Giving Amount UofU": "num",
        "Has UofU Legacy Gift": "bool", "Legacy Circle Member PBS Utah": "bool",
        "Sustainer PBS Utah": "bool",
    },
    "unite_payments": {"credit_date": "date", "payment_amount": "num"},
    "soft_credits": {"Credit Date": "date", "Credit Amount": "num"},
    "team_approach_legacy_payments": {"gift_date": "date", "payment_amount": "num"},
    "campaign_codes": {"Active": "bool", "Solicitation": "bool"},
    "campaign_members": {
        "responded": "bool", "indirect_response": "bool",
        "responded_date": "date", "campaign_start_date": "date",
    },
    "passport_viewing": {"fiscal_year": "int", "titles_viewed": "int", "mean_percent_watched": "num"},
    "cultivation": {
        "fiscal_year": "int", "selected": "bool", "contacted": "bool",
        "holdout": "bool", "program_adopted": "bool",
    },
    "officer_portfolios": {"capacity": "int", "rollout_fy": "num"},
}


# ---------------------------------------------------------------- loading

def _snake(name):
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def _cast(col, kind):
    c = pl.col(col)
    if kind == "date":
        return c.str.to_date("%Y-%m-%d", strict=False)
    if kind == "num":
        return c.cast(pl.Float64, strict=False)
    if kind == "int":
        return c.cast(pl.Float64, strict=False).cast(pl.Int64)
    return pl.when(c.is_null()).then(None).otherwise(c == "True").alias(col)


def load_tables(raw_dir, cache_dir):
    """Read each CSV as text, cast the typed columns, snake-case the names, and cache as Parquet."""
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, types in TYPES.items():
        out = cache_dir / f"{name}.parquet"
        if not out.exists():
            df = pl.read_csv(raw_dir / f"{name}.csv", infer_schema_length=0)
            df = df.with_columns([_cast(c, k) for c, k in types.items()])
            df.rename({c: _snake(c) for c in df.columns}).write_parquet(out)
        tables[name] = pl.read_parquet(out)
    return tables


def fiscal_year(col):
    """PBS Utah fiscal years run July to June, so July 2020 is FY2021."""
    return pl.col(col).dt.year() + (pl.col(col).dt.month() >= 7).cast(pl.Int32)


# ---------------------------------------------------------------- cleaning (decisions 1, 4, 6)

def _flag_copies(df, row_id, checks):
    """A row is a copy when it fails a check and an otherwise identical valid row exists.

    checks maps a check name to (expression that is True when the row fails, the field that
    check is about). "Otherwise identical" means equal on every column except the row ID and
    the failing field.
    """
    df = df.with_columns([expr.alias(name) for name, (expr, _) in checks.items()])
    valid = df.filter(~pl.any_horizontal([pl.col(name) for name in checks]))
    out = []
    for name, (_, field) in checks.items():
        match_on = [c for c in df.columns if c not in (row_id, field, *checks)]
        failing = df.filter(pl.col(name))
        twins = failing.join(valid.select(match_on).unique(), on=match_on, how="semi", nulls_equal=True)
        out.append(failing.select(row_id, check=pl.lit(name),
                                  has_valid_twin=pl.col(row_id).is_in(twins[row_id].implode())))
    return pl.concat(out)


def clean_tables(t):
    """Apply the shared cleaning rules. Returns the cleaned tables and an audit of what was removed."""
    con, pay, sc, ta, cc = (t["constituents_w_memberships"], t["unite_payments"], t["soft_credits"],
                            t["team_approach_legacy_payments"], t["campaign_codes"])
    ids = con["constituent_id"].implode()
    codes = cc["marketing_code"].implode()

    # Unite payments: broken donor link, broken campaign link, or a non-positive amount
    pay_flags = _flag_copies(pay, "payment_id", {
        "donor is not a constituent": (~pl.col("donor_id").is_in(ids), "donor_id"),
        "campaign code not in campaign table": (~pl.col("marketing_code").is_in(codes), "marketing_code"),
        "amount of $0 or less": (pl.col("payment_amount") <= 0, "payment_amount"),
    })

    # soft credits: exact duplicates first, then broken recipient or non-positive amount
    sc_dedup = sc.unique(maintain_order=True)
    sc_flags = _flag_copies(sc_dedup, "credit_id", {
        "recipient is not a constituent": (~pl.col("soft_credit_constituent_id").is_in(ids), "soft_credit_constituent_id"),
        "amount of $0 or less": (pl.col("credit_amount") <= 0, "credit_amount"),
    })

    # legacy: rows with no linked constituent that repeat a linked row on every non-ID column
    linked = ta.filter(pl.col("constituent_id_1").is_not_null())
    unlinked = ta.filter(pl.col("constituent_id_1").is_null())
    legacy_on = [c for c in ta.columns if c != "ta_account_id" and not c.startswith("constituent_id")]
    unlinked_copies = unlinked.join(linked.select(legacy_on).unique(), on=legacy_on, how="semi", nulls_equal=True)

    pay_copy_ids = pay_flags.filter("has_valid_twin")["payment_id"].unique()
    sc_copy_ids = sc_flags.filter("has_valid_twin")["credit_id"].unique()

    audit = {
        "pay_flags": pay_flags,
        "sc_flags": sc_flags,
        "sc_exact_duplicates": sc.height - sc_dedup.height,
        "legacy_unlinked": unlinked.height,
        "legacy_unlinked_copies": unlinked_copies.height,
        "pay_copy_ids": pay_copy_ids,
        "sc_copy_ids": sc_copy_ids,
        "legacy_copy_ids": unlinked_copies["ta_account_id"],
    }
    clean = {
        "constituents": con,
        "payments": pay.filter(~pl.col("payment_id").is_in(pay_copy_ids.implode())),
        "soft_credits": sc_dedup.filter(~pl.col("credit_id").is_in(sc_copy_ids.implode())),
        "legacy": linked,                              # unlinked rows are all copies
        "campaign_members": t["campaign_members"],     # repeat solicitations are kept
        "campaign_codes": cc,
        "passport": t["passport_viewing"],
        "cultivation": t["cultivation"],
        "portfolios": t["officer_portfolios"],
    }
    return clean, audit


def label_agreement(t, audit):
    """Compare the rows the content rules removed with the copy labels in the row IDs."""
    def check(table, all_ids, removed, prefix):
        labeled = all_ids.filter(all_ids.str.starts_with(prefix))
        return {
            "table": table, "copy_label": prefix,
            "removed_by_rules": removed.len(), "carry_copy_label": labeled.len(),
            "labeled_but_kept": (~labeled.is_in(removed.implode())).sum(),
            "removed_but_unlabeled": (~removed.is_in(labeled.implode())).sum(),
        }
    return pl.DataFrame([
        check("unite_payments", t["unite_payments"]["payment_id"], audit["pay_copy_ids"], "SYN-DEF-"),
        check("soft_credits", t["soft_credits"].unique()["credit_id"], audit["sc_copy_ids"], "SYN-DEF-"),
        check("team_approach_legacy_payments", t["team_approach_legacy_payments"]["ta_account_id"],
              audit["legacy_copy_ids"], "SYN-UNMATCHED"),
    ])


# ---------------------------------------------------------------- panel (decisions 4, 15)

def organization_ids(con):
    """Organizations are identified by their constituent type, not by their ID pattern."""
    return con.filter(pl.col("primary_constituent_type") == "Organization")["constituent_id"]


def build_person_years(clean):
    """One row per person per fiscal year with giving, sources, and major status under the household rule."""
    con = clean["constituents"]
    orgs = organization_ids(con).implode()

    unite = (
        clean["payments"].filter(~pl.col("donor_id").is_in(orgs))
        .group_by(constituent_id=pl.col("donor_id"), fy=fiscal_year("credit_date"))
        .agg(
            unite_amt=pl.col("payment_amount").sum(),
            unite_payments=pl.len(),
            recurring_amt=pl.col("payment_amount").filter(pl.col("pledge_gift_type") == "Recurring").sum(),
            recurring_payments=(pl.col("pledge_gift_type") == "Recurring").sum(),
            outright_amt=pl.col("payment_amount").filter(pl.col("pledge_gift_type") == "Outright").sum(),
            max_payment=pl.col("payment_amount").max(),
            upgrade_gift=(pl.col("gift_type") == "Upgrade").any(),
            additional_gift=(pl.col("gift_type") == "Additional").any(),
        )
    )
    soft = (
        clean["soft_credits"]
        .group_by(constituent_id=pl.col("soft_credit_constituent_id"), fy=fiscal_year("credit_date"))
        .agg(soft_amt=pl.col("credit_amount").sum(), soft_credits=pl.len())
    )
    legacy = (
        clean["legacy"]
        .group_by(constituent_id=pl.col("constituent_id_1"), fy=fiscal_year("gift_date"))
        .agg(
            legacy_amt=pl.col("payment_amount").sum(),
            legacy_gifts=pl.len(),
            shared_legacy_amt=pl.col("payment_amount").filter(pl.col("constituent_id_2").is_not_null()).sum(),
        )
    )
    py = (
        pl.concat([unite, soft, legacy], how="diagonal")
        .group_by("constituent_id", "fy")
        .agg(
            pl.col("unite_amt", "unite_payments", "recurring_amt", "recurring_payments", "outright_amt",
                   "soft_amt", "soft_credits", "legacy_amt", "legacy_gifts", "shared_legacy_amt").sum(),
            pl.col("max_payment").max(),
            pl.col("upgrade_gift", "additional_gift").any(),
        )
        .with_columns(
            total=pl.sum_horizontal("unite_amt", "soft_amt", "legacy_amt"),
            upgrade_gift=pl.col("upgrade_gift").fill_null(False),
            additional_gift=pl.col("additional_gift").fill_null(False),
        )
    )

    # household rule: attach the spouse's total in the same year
    spouses = con.filter(pl.col("spouse_constituent_id").is_not_null()).select(
        "constituent_id", spouse_id=pl.col("spouse_constituent_id")
    )
    py = (
        py.join(spouses, on="constituent_id", how="left")
        .join(py.select(spouse_id=pl.col("constituent_id"), fy="fy", spouse_total="total"),
              on=["spouse_id", "fy"], how="left")
        .with_columns(spouse_total=pl.col("spouse_total").fill_null(0))
        .with_columns(
            major_individual=pl.col("total") >= MAJOR,
            major_combined_only=(pl.col("total") < MAJOR) & (pl.col("spouse_total") < MAJOR)
                                & (pl.col("total") + pl.col("spouse_total") >= MAJOR)
                                & pl.col("spouse_id").is_not_null(),
            # a couple counts once; a person without a spouse is their own unit
            couple_id=pl.when(pl.col("spouse_id").is_not_null())
                        .then(pl.min_horizontal("constituent_id", "spouse_id"))
                        .otherwise(pl.col("constituent_id")),
        )
        .with_columns(major=pl.col("major_individual") | pl.col("major_combined_only"))
        .with_columns(tier=pl.col("total").cut(TIERS, labels=TIER_LABELS, left_closed=True).cast(pl.Enum(TIER_LABELS)))
        .sort("constituent_id", "fy")
    )
    return py


def individual_gifts(clean):
    """Every gift credited to a person, with its date: own payments, soft credits, and legacy gifts."""
    orgs = organization_ids(clean["constituents"]).implode()
    return pl.concat([
        clean["payments"].filter(~pl.col("donor_id").is_in(orgs))
        .select(constituent_id=pl.col("donor_id"), date=pl.col("credit_date"), amount=pl.col("payment_amount")),
        clean["soft_credits"]
        .select(constituent_id=pl.col("soft_credit_constituent_id"), date=pl.col("credit_date"), amount=pl.col("credit_amount")),
        clean["legacy"]
        .select(constituent_id=pl.col("constituent_id_1"), date=pl.col("gift_date"), amount=pl.col("payment_amount")),
    ])


def giving_features(rows, py, gifts):
    """Giving-history features at the end of FY t for each (constituent_id, t) in rows.

    Everything is measured from gifts dated on or before June 30 of FY t, so nothing comes
    from the outcome years.
    """
    keys = rows.select("constituent_id", "t").unique()
    hist = keys.join(py, on="constituent_id", how="inner").filter(pl.col("fy") <= pl.col("t"))
    lag = pl.col("t") - pl.col("fy")
    history = hist.group_by("constituent_id", "t").agg(
        prev_total=pl.col("total").filter(lag == 1).sum(),
        two_years_before=pl.col("total").filter(lag.is_between(1, 2)).sum(),
        years_last5=pl.col("fy").filter(lag <= 4).n_unique(),
        first_fy=pl.col("fy").min(),
        legacy_years=pl.col("fy").filter(pl.col("legacy_gifts") > 0).n_unique(),
        legacy_gift_count=pl.col("legacy_gifts").sum(),
        soft_before=(pl.col("soft_amt").filter(lag >= 1) > 0).any(),
        soft_in_t=(pl.col("soft_amt").filter(lag == 0) > 0).any(),
        upgrade_gift=pl.col("upgrade_gift").filter(lag == 0).any(),
        additional_gift=pl.col("additional_gift").filter(lag == 0).any(),
        recurring_amt=pl.col("recurring_amt").filter(lag == 0).sum(),
        recurring_payments=pl.col("recurring_payments").filter(lag == 0).sum(),
    )
    g = keys.join(gifts, on="constituent_id", how="inner").with_columns(
        end=pl.date(pl.col("t"), 6, 30), fy=fiscal_year("date")
    ).filter(pl.col("date") <= pl.col("end"))
    last = g.group_by("constituent_id", "t").agg(
        last_gift=pl.col("date").max(),
        largest_gift_t=pl.col("amount").filter(pl.col("fy") == pl.col("t")).max(),
        end=pl.col("end").first(),
    )
    return (
        keys.join(history, on=["constituent_id", "t"], how="left")
        .join(last, on=["constituent_id", "t"], how="left")
        .with_columns(
            days_since_last_gift=(pl.col("end") - pl.col("last_gift")).dt.total_days(),
            years_since_first_gift=pl.col("t") - pl.col("first_fy") + 1,
            sustainer=pl.col("recurring_payments").fill_null(0) > 0,
            monthly_amount=pl.when(pl.col("recurring_payments") > 0)
                             .then(pl.col("recurring_amt") / pl.col("recurring_payments")),
        )
        .drop("end", "last_gift", "first_fy")
    )


CHANNELS = {"mail": ["Mail"], "email": ["Email"], "digital": ["Digital"], "phone_or_in_person": ["Phone", "In-Person"]}


def solicitation_by_year(clean):
    """Solicitations, responses, and channels per person per fiscal year.

    A response counts in the fiscal year of its response date, so it is known by the end
    of that year (Samantha's timing rule). Campaign start dates are fiscal-year markers.
    """
    cm = clean["campaign_members"].join(
        clean["campaign_codes"].select("marketing_code", "communication_method"), on="marketing_code", how="left"
    )
    sent = cm.group_by("constituent_id", fy=fiscal_year("campaign_start_date")).agg(
        solicitations=pl.len(),
        **{f"by_{k}": pl.col("communication_method").is_in(v).any() for k, v in CHANNELS.items()},
    )
    answered = cm.filter(pl.col("responded")).group_by("constituent_id", fy=fiscal_year("responded_date")).agg(
        responses=pl.len()
    )
    return sent.join(answered, on=["constituent_id", "fy"], how="full", coalesce=True).with_columns(
        pl.col("solicitations", "responses").fill_null(0), pl.col("^by_.*$").fill_null(False)
    )


def solicitation_features(rows, sol):
    """Solicitation features at the end of FY t: FY t counts and channels, plus a three-year count."""
    keys = rows.select("constituent_id", "t").unique()
    this_year = sol.rename({"fy": "t"})
    three_years = (
        keys.join(sol, on="constituent_id", how="inner")
        .filter(pl.col("fy").is_between(pl.col("t") - 2, pl.col("t")))
        .group_by("constituent_id", "t").agg(solicitations_3yr=pl.col("solicitations").sum())
    )
    return (
        keys.join(this_year, on=["constituent_id", "t"], how="left")
        .join(three_years, on=["constituent_id", "t"], how="left")
        .with_columns(pl.col("solicitations", "responses", "solicitations_3yr").fill_null(0),
                      pl.col("^by_.*$").fill_null(False))
    )


def passport_features(rows, passport):
    """Passport viewing at the end of FY t. A missing year means no viewing record, kept as its own level."""
    keys = rows.select("constituent_id", "t").unique()
    by_year = passport.group_by("constituent_id", t=pl.col("fiscal_year")).agg(
        titles=pl.col("titles_viewed").sum(),
        genres=pl.len(),
        share_watched=(pl.col("mean_percent_watched") * pl.col("titles_viewed")).sum() / pl.col("titles_viewed").sum(),
    )
    first_view = passport.group_by("constituent_id").agg(first_view_fy=pl.col("fiscal_year").min())
    return (
        keys.join(by_year, on=["constituent_id", "t"], how="left")
        .join(first_view, on="constituent_id", how="left")
        .with_columns(
            viewed_t=pl.col("titles").is_not_null(),
            viewed_by_t=pl.col("first_view_fy").is_not_null() & (pl.col("first_view_fy") <= pl.col("t")),
        )
        .drop("first_view_fy")
    )


# ---------------------------------------------------------------- cohorts (decisions 2, 3, 5, 9)

def build_cohorts(py, windows=(1, 3, 5), first=FIRST_COHORT):
    """Eligible and lapsed donor rows at the end of each FY t, with outcomes for each window.

    in_validation marks cohorts whose outcome years all end before the funding cut, so the
    FY2025 cohort (outcomes in FY2026) is kept for final training only (decision 9).
    """
    last_fy = int(py["fy"].max())
    majors = py.filter(pl.col("major")).select("constituent_id", "fy")
    gave = py.select("constituent_id", "fy", "total")
    rows = []
    for w in windows:
        for t in range(first, last_fy - w + 1):
            former = majors.filter(pl.col("fy") < t)["constituent_id"].unique().implode()
            converted = majors.filter(pl.col("fy").is_between(t + 1, t + w))["constituent_id"].unique().implode()
            major_t = majors.filter(pl.col("fy") == t)["constituent_id"].implode()

            active = (
                py.filter((pl.col("fy") == t) & ~pl.col("major"))
                .select("constituent_id", "couple_id", prior_total=pl.col("total"), segment=pl.lit("active"))
            )
            gave_recent = gave.filter(pl.col("fy").is_between(t - 2, t - 1))["constituent_id"].unique()
            lapsed = (
                pl.DataFrame({"constituent_id": gave_recent})
                .join(gave.filter(pl.col("fy") == t), on="constituent_id", how="anti")
                .filter(~pl.col("constituent_id").is_in(major_t))
                .join(py.select("constituent_id", "couple_id").unique("constituent_id"), on="constituent_id", how="left")
                .with_columns(prior_total=pl.lit(0.0), segment=pl.lit("lapsed"))
            )
            rows.append(
                pl.concat([active, lapsed], how="diagonal_relaxed").with_columns(
                    former_major=pl.col("constituent_id").is_in(former),
                    converted=pl.col("constituent_id").is_in(converted),
                    t=pl.lit(t), window=pl.lit(w),
                    in_validation=pl.lit(t + w < FUNDING_CUT_FY),
                )
            )
    return pl.concat(rows)
