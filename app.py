import csv
import datetime
from pathlib import Path

import streamlit as st
from engine import FAQBot, THRESHOLD


LOG = Path("referral_log.csv")


st.set_page_config(
    page_title="ShopEase Support Copilot",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Customer Support Reply Drafter")

st.caption(
    "Drafts replies using FAQs and Generative AI. "
    "Unknown questions go to a human."
)


@st.cache_resource
def load_bot():
    return FAQBot()


bot = load_bot()


threshold = st.sidebar.slider(
    "Referral threshold",
    0.05,
    0.60,
    THRESHOLD,
    0.05,
    help="Lower = answers more, Higher = refers more"
)

st.sidebar.markdown(
    "**FAQs loaded:** " + str(len(bot.faqs))
)


query = st.text_area(
    "Customer question",
    placeholder="e.g. How long does delivery take?"
)


def log_referral(q, reason):
    new = not LOG.exists()

    with LOG.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        if new:
            writer.writerow(["time", "query", "reason"])

        writer.writerow([
            datetime.datetime.now().isoformat(
                timespec="seconds"
            ),
            q,
            reason
        ])


# Generate response
if st.button("Draft reply", type="primary"):

    if not query.strip():
        st.warning("Please enter a customer question first.")

    else:
        try:
            with st.spinner("Generating your response..."):

                st.session_state["res"] = bot.respond(
                    query,
                    threshold
                )

                st.session_state["manual"] = False

        except Exception as e:
            st.error(f"Unable to generate reply: {e}")


# Display result
res = st.session_state.get("res")


if res:

    referred = (
        res.status == "REFER_TO_HUMAN"
        or st.session_state.get("manual", False)
    )

    if referred:

        st.error("🔴 REFER TO HUMAN AGENT")

        reason = (
            "Manual override by reviewer"
            if st.session_state.get("manual", False)
            else res.reason
        )

        st.write("**Reason:**", reason)

        st.text_area(
            "Holding message to customer",
            res.draft or (
                "Thanks for reaching out. "
                "Our support team will contact you shortly."
            ),
            height=140
        )

        if st.button("Log this referral"):

            log_referral(res.query, reason)

            st.success("Logged to referral_log.csv")

    else:

        st.success("🟢 RESPONSE DRAFTED")

        # Show actual response source
        if res.source == "Gemini AI":

            st.success(
                "✨ Response generated using Gemini AI"
            )

        elif res.source == "FAQ Template":

            st.warning(
                "📝 Template fallback used "
                "(Gemini unavailable)"
            )

        st.text_area(
            "Draft reply (edit before sending)",
            res.draft,
            height=200
        )

        if st.button("⚠️ Refer to human instead"):

            st.session_state["manual"] = True

            st.rerun()


    st.progress(
        min(res.confidence, 1.0),
        text=f"Confidence: {res.confidence:.2f}"
    )

    with st.expander("Top matching FAQs"):

        for faq, score in res.matches:

            st.markdown(
                f"**{faq['question']}** (score {score:.2f})"
            )

            st.caption(faq["answer"])
