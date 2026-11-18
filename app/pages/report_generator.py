"""Generate downloadable historical air-quality PDF reports for EcoPulse."""

from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether


DARK = colors.HexColor("#0B3B44")
TEAL = colors.HexColor("#0E7C86")
MINT = colors.HexColor("#B9E6D4")
IVORY = colors.HexColor("#F4F8F3")
MUTED = colors.HexColor("#63867C")


def generate_air_report(data, city=None, start_date=None, end_date=None):
    """Return PDF bytes summarizing filtered historical pollution observations.

    Args:
        data: DataFrame with Date, City and pollutant columns.
        city: Optional city name; None or 'All Cities' includes all cities.
        start_date: Optional inclusive lower date bound.
        end_date: Optional inclusive upper date bound.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    required = {"Date", "City", "PM2.5"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "City"])
    if city and city != "All Cities":
        df = df.loc[df["City"].astype(str) == str(city)]
    if start_date is not None:
        df = df.loc[df["Date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df.loc[df["Date"] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    if df.empty:
        raise ValueError("No historical records match the selected filters.")

    pollutants = [p for p in ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"] if p in df.columns]
    for pollutant in pollutants:
        df[pollutant] = pd.to_numeric(df[pollutant], errors="coerce")

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=(595.28, 841.89), rightMargin=48, leftMargin=48,
        topMargin=50, bottomMargin=48, title="EcoPulse | Historical Air Quality Report",
        author="EcoPulse",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="EcoTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=DARK, spaceAfter=7))
    styles.add(ParagraphStyle(name="EcoSubtitle", parent=styles["Normal"], fontSize=10, leading=15, textColor=MUTED, spaceAfter=18))
    styles.add(ParagraphStyle(name="EcoSection", parent=styles["Heading2"], fontSize=12, leading=16, textColor=DARK, spaceBefore=18, spaceAfter=9))
    styles.add(ParagraphStyle(name="EcoBody", parent=styles["Normal"], fontSize=9, leading=14, textColor=DARK))
    styles.add(ParagraphStyle(name="EcoSmall", parent=styles["Normal"], fontSize=8, leading=12, textColor=MUTED))
    story = [
        Paragraph("EcoPulse", styles["EcoTitle"]),
        Paragraph("HISTORICAL AIR QUALITY REPORT  |  India", styles["EcoSubtitle"]),
    ]

    date_min = df["Date"].min().strftime("%d %b %Y")
    date_max = df["Date"].max().strftime("%d %b %Y")
    city_count = df["City"].nunique()
    city_label = str(city) if city and city != "All Cities" else "All available cities"
    summary = [
        ["Coverage", city_label],
        ["Observation period", f"{date_min} - {date_max}"],
        ["Records", f"{len(df):,}"],
        ["Cities represented", str(city_count)],
    ]
    summary_table = Table(summary, colWidths=[150, 345], hAlign="LEFT")
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), IVORY),
        ("TEXTCOLOR", (0, 0), (0, -1), DARK),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#D8E9DF")),
    ]))
    story.extend([Paragraph("Report overview", styles["EcoSection"]), summary_table])

    def fmt(value):
        return "N/A" if pd.isna(value) else f"{value:,.2f}"

    rows = [["Pollutant", "Mean", "Minimum", "Maximum", "Valid records"]]
    for pollutant in pollutants:
        s = df[pollutant].dropna()
        rows.append([pollutant, fmt(s.mean()), fmt(s.min()), fmt(s.max()), f"{len(s):,}"])
    if len(rows) > 1:
        table = Table(rows, colWidths=[100, 90, 90, 90, 125], repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, IVORY]),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ("LINEBELOW", (0, -1), (-1, -1), 0.7, MINT),
        ]))
        story.extend([Paragraph("Pollutant summary", styles["EcoSection"]), table])

    pm25 = df["PM2.5"].dropna()
    story.append(Paragraph("PM2.5 observations", styles["EcoSection"]))
    if pm25.empty:
        story.append(Paragraph("No valid PM2.5 measurements were available in the selected records.", styles["EcoBody"]))
    else:
        story.append(Paragraph(
            f"The mean recorded PM2.5 concentration was <b>{fmt(pm25.mean())}</b> "
            f"micrograms per cubic metre (µg/m³), based on {len(pm25):,} valid observations. "
            f"The lowest and highest recorded values were {fmt(pm25.min())} and {fmt(pm25.max())} µg/m³, respectively.",
            styles["EcoBody"],
        ))

    if city_count > 1:
        ranked = (df.groupby("City")["PM2.5"].mean().dropna().sort_values(ascending=False).head(10))
        if not ranked.empty:
            rank_rows = [["City", "Average PM2.5 (µg/m³)"]]
            rank_rows += [[str(name), fmt(value)] for name, value in ranked.items()]
            ranking = Table(rank_rows, colWidths=[270, 225], repeatRows=1, hAlign="LEFT")
            ranking.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, IVORY]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]))
            story.extend([Paragraph("Highest city averages (selected data)", styles["EcoSection"]), ranking])

    story.extend([
        Spacer(1, 18),
        Paragraph("Data notes", styles["EcoSection"]),
        Paragraph(
            "This report summarizes the historical observations provided to EcoPulse. "
            "Results depend on dataset coverage, missing measurements, and selected filters. "
            "Averages are descriptive statistics, not forecasts or live air-quality readings. "
            "Pollutant values are shown in the units supplied by the dataset; PM2.5 is expressed in µg/m³.",
            styles["EcoSmall"],
        ),
    ])

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(MINT)
        canvas.line(48, 39, 547, 39)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(48, 27, "EcoPulse  |  Historical air pollution analysis")
        canvas.drawRightString(547, 27, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
