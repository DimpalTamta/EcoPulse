"""PDF summary for the water-quality data."""
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from water_data import CLASS_ORDER, class_counts, share_good

DARK = colors.HexColor("#0B3B44")


def generate_water_report(df, state, y0, y1):
    d = df[df["Year"].between(y0, y1)]
    if state != "All states":
        d = d[d["State"] == state]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=48, rightMargin=48,
                            topMargin=50, bottomMargin=48,
                            title="EcoPulse | Water Quality Report", author="EcoPulse")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=DARK, fontSize=24)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=DARK)
    body = ss["BodyText"]

    def table(rows, widths=None):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8F3")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CFE3D8")),
        ]))
        return t

    c = class_counts(d)
    tot = max(int(c.sum()), 1)
    story = [
        Paragraph("EcoPulse", h1),
        Paragraph(f"Water quality summary · {state} · {y0}–{y1}", h2),
        Paragraph(
            f"{len(d):,} station-year records from {d['Station Code'].nunique():,} stations. "
            f"{share_good(d):.0f}% of classifiable records meet at least Class C.", body),
        Spacer(1, 12),
        Paragraph("Screening class mix", h2),
        table([["Class", "Records", "Share"]] +
              [[k, f"{int(c[k]):,}", f"{c[k] / tot * 100:.0f}%"] for k in CLASS_ORDER]),
        Spacer(1, 12),
        Paragraph("Key medians", h2),
        table([["Parameter", "Median"]] + [
            [n, f"{d[col].median():,.2f}"] for n, col in [
                ("Dissolved oxygen (mg/L)", "Dissolved oxygen"), ("BOD (mg/L)", "BOD"),
                ("pH", "pH"), ("Conductivity (µmho/cm)", "Conductivity"),
                ("Fecal coliform (MPN/100ml)", "Fecal coliform")]]),
    ]
    if state == "All states":
        g = d.groupby("State").agg(Records=("Year", "size"), BOD=("BOD", "median")).reset_index()
        g = g[g["Records"] >= 10].sort_values("BOD", ascending=False).head(10)
        story += [Spacer(1, 12), Paragraph("States with highest median BOD", h2),
                  table([["State", "Records", "Median BOD"]] +
                        [[r.State, str(r.Records), f"{r.BOD:.1f}"] for r in g.itertuples()])]
    story += [Spacer(1, 14), Paragraph(
        "Classes are simplified CPCB-style screening labels on mid-point values. "
        "This report is for education and is not an official assessment.", body)]
    doc.build(story)
    return buf.getvalue()
