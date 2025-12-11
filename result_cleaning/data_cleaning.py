import streamlit as st
import pandas as pd
import math
import os

# --- Configuration ---
ROWS_PER_PAGE = 500
SCAM_CATEGORIES = [
    "Account or Identity Takeover Scams",
    "Buying and Selling (Products or Services) Scams",
    "Donation Scams",
    "Investment Scams",
    "Jobs and Employment Scams",
    "Money Recovery Scams",
    "Payment Redirection/ Business Email Compromise Scams",
    "Phishing Scams",
    "Relationship Scams",
    "Threat Scams",
    "Unexpected Money Scams",
    "Other Scams",
]


def main():
    st.set_page_config(
        layout="wide",
        page_title="Scam Classifier Review",
        initial_sidebar_state="collapsed",
    )

    # --- Sidebar: File Settings ---
    with st.sidebar:
        st.header("📂 File Settings")

        # 1. Input File Selector
        uploaded_file = st.file_uploader("Upload CSV to Review", type=["csv"])

        st.divider()

        # 2. Output File Name
        output_filename = st.text_input(
            "Output Filename",
            value="updated_training.csv",
            help="The name of the file to save your results to.",
        )

        # Ensure it ends in .csv
        if not output_filename.endswith(".csv"):
            output_filename += ".csv"

    # --- Main Area ---
    st.title("🛡️ Scam Classification Reviewer")

    if uploaded_file is None:
        st.info("👈 Please upload a CSV file in the sidebar to begin.")
        st.markdown(
            """
        **Expected CSV Format:**
        * `Description` (Text of the scam)
        * `Predicted Category` (The AI's guess)
        * `Confidence` (Score)
        """
        )
        return

    # --- Unique Key for Session State ---
    file_key = f"data_{uploaded_file.name}"

    # --- DATA LOADING ---
    if file_key not in st.session_state:
        try:
            df = pd.read_csv(uploaded_file)

            # Validation
            required_cols = {"Description", "Predicted Category"}
            if not required_cols.issubset(df.columns):
                st.error(
                    f"Error: CSV is missing columns: {required_cols - set(df.columns)}"
                )
                return

            # Add columns if missing
            if "irrelevant" not in df.columns:
                df["irrelevant"] = False

            # Add 'Thumbs Up' / Verified column (always ensure it's first later via reordering if needed)
            # We use a special symbol column name
            if "👍" not in df.columns:
                df.insert(0, "👍", False)  # Insert at position 0

            # Store in session state
            st.session_state[file_key] = df
            st.session_state["page_number"] = 0

            # Cleanup old keys
            for k in list(st.session_state.keys()):
                if k.startswith("data_") and k != file_key:
                    del st.session_state[k]

        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

    # --- Retrieve Data ---
    full_df = st.session_state[file_key]

    # --- PAGINATION LOGIC ---
    total_rows = len(full_df)
    total_pages = math.ceil(total_rows / ROWS_PER_PAGE)

    if "page_number" not in st.session_state:
        st.session_state["page_number"] = 0

    current_page = st.session_state["page_number"]

    # Slice for current page
    start_idx = current_page * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE
    current_page_df = full_df.iloc[start_idx:end_idx].copy()

    # --- SIDEBAR STATS & NAVIGATION ---
    with st.sidebar:
        st.divider()
        st.markdown(f"**Total Rows:** {total_rows}")
        st.markdown(f"**Page:** {current_page + 1} of {total_pages}")
        st.markdown(f"**Showing:** {start_idx + 1} - {min(end_idx, total_rows)}")

        col_prev, col_next = st.columns(2)

        # We need a placeholder to detect if we can switch pages
        # The switch happens AFTER the editor is rendered and we check for changes

        # We use buttons with callbacks or standard buttons + logic?
        # Logic: If button clicked, check 'editor_state'. If dirty, warn. Else, change page state.
        # Limitation: 'st.data_editor' output is only available 'below'.
        # Solution: We render the editor first? No, we need stats first.
        # Workaround: We will use a flag 'can_navigate' determined by the PREVIOUS run's check?
        # Actually, simpler: We will use a flag 'page_is_dirty' calculated after the editor.
        pass

    # --- DATA EDITOR ---
    st.markdown(f"#### Page {current_page + 1}")

    edited_df = st.data_editor(
        current_page_df,
        column_config={
            "👍": st.column_config.CheckboxColumn(
                "👍",
                help="Check to mark as correct/verified",
                default=False,
                width="small",
            ),
            "Description": st.column_config.TextColumn(
                "Description", disabled=True, width=600
            ),
            "Predicted Category": st.column_config.SelectboxColumn(
                "Category", options=SCAM_CATEGORIES, required=True, width="medium"
            ),
            "Confidence": st.column_config.NumberColumn(
                "Conf.", format="%.2f", disabled=True, width="small"
            ),
            "irrelevant": st.column_config.CheckboxColumn("Irrelevant?", default=False),
        },
        hide_index=True,
        # Reorder to ensure 👍 is first
        column_order=[
            "👍",
            "Description",
            "Predicted Category",
            "Confidence",
            "irrelevant",
        ],
        num_rows="fixed",
        height=600,
        key=f"editor_page_{current_page}",  # vital to reset state on page switch
    )

    # --- CHANGE DETECTION ---
    # We compare edited_df vs current_page_df
    # We need to fillna to safely compare
    df_diff = edited_df.fillna("MAGIC_NAN") != current_page_df.fillna("MAGIC_NAN")

    # 'changed_rows' mask: True if any column changed
    # Logic: "modified" means content changed. "verified" means 👍 is checked.
    # We save if (Modified) OR (Verified).

    # Check if ANY column changed for each row
    row_modified_mask = df_diff.any(axis=1)

    # Check if 👍 is true
    row_verified_mask = edited_df["👍"] == True

    # Combined mask for "Needs Saving"
    rows_to_save_mask = row_modified_mask | row_verified_mask

    num_modified = row_modified_mask.sum()
    num_to_save = rows_to_save_mask.sum()

    # Is the page "dirty"? (Unsaved changes)
    # Dirty if modified but NOT saved yet.
    # In Streamlit, 'edited_df' reflects current state.
    # We can assume if 'num_modified > 0', we have potential changes to save.

    page_is_dirty = num_to_save > 0

    # --- SIDEBAR NAV ACTIONS (Post-Calculation) ---
    # We do this here because we needed 'page_is_dirty' status.
    with st.sidebar:
        st.divider()
        if st.button("Previous Page", disabled=(current_page == 0)):
            if page_is_dirty:
                st.warning(
                    "⚠️ You have unsaved changes on this page! Save or revert before switching."
                )
            else:
                st.session_state["page_number"] -= 1
                st.rerun()

        if st.button("Next Page", disabled=(current_page >= total_pages - 1)):
            if page_is_dirty:
                st.warning(
                    "⚠️ You have unsaved changes on this page! Save or revert before switching."
                )
            else:
                st.session_state["page_number"] += 1
                st.rerun()

    # --- SAVE MECHANISM ---
    st.divider()

    # Feedback metrics
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("📝 Modified Rows", num_modified)
    m_col2.metric("👍 Verified Rows", row_verified_mask.sum())
    m_col3.metric("💾 Total to Save", num_to_save)

    if st.button(
        f"Save Changes to '{output_filename}'",
        type="primary",
        disabled=(num_to_save == 0),
    ):
        try:
            # --- VALIDATION ---
            # Check Verified Rows: Must have Category OR be Irrelevant
            # We look at edited_df rows where 👍 is True
            verified_rows = edited_df[edited_df["👍"] == True]
            # Identify invalid ones: Irrelevant is False AND Category is missing/empty
            # Note: Category column is 'Predicted Category'
            invalid_verified = verified_rows[
                (verified_rows["irrelevant"] == False)
                & (
                    verified_rows["Predicted Category"].isna()
                    | (verified_rows["Predicted Category"] == "")
                )
            ]

            if len(invalid_verified) > 0:
                st.error(
                    f"❌ Cannot save: {len(invalid_verified)} row(s) marked as Verified (👍) have no Category. Please select a Category or mark as Irrelevant."
                )
                return

            # 1. Extract rows to save
            df_to_save = edited_df[rows_to_save_mask].copy()

            # 2. Filter logic (Irrelevant / Empty Category)
            # Remove irrelevant
            df_to_save = df_to_save[df_to_save["irrelevant"] == False]
            # Remove empty category
            df_to_save = df_to_save[df_to_save["Predicted Category"].notna()]
            df_to_save = df_to_save[df_to_save["Predicted Category"] != ""]

            # Rename for output if needed
            if "Predicted Category" in df_to_save.columns:
                df_to_save = df_to_save.rename(
                    columns={"Predicted Category": "Category"}
                )

            # Clean up columns for output (remove 👍)
            final_cols = ["Description", "Category", "irrelevant"]
            save_ready_df = df_to_save[final_cols]

            # 3. Append & Deduplicate
            if os.path.exists(output_filename):
                try:
                    existing_df = pd.read_csv(output_filename)
                    combined_df = pd.concat(
                        [existing_df, save_ready_df], ignore_index=True
                    )
                    save_ready_df = combined_df.drop_duplicates(
                        subset=["Description"], keep="last"
                    )
                except Exception as e:
                    st.warning(f"Error reading existing file: {e}. Overwriting.")

            if len(save_ready_df) > 0:
                save_ready_df.to_csv(output_filename, index=False)
                st.success(
                    f"✅ Saved {len(save_ready_df)} rows to **{output_filename}**"
                )

                # 4. UPDATE SESSION STATE (Sync)
                # We must update the 'full_df' in session state with the edits
                # so that the "Dirty" check clears.
                # 'edited_df' has the new values. We put them back into 'full_df'.

                # Map the edits back to full_df using index
                # Note: 'current_page_df' has the original indices of 'full_df' preserved?
                # 'iloc' slicing does NOT preserve original index if we didn't reset it?
                # Actually, 'full_df' has an index. 'current_page_df' is a slice, so it keeps the index.
                # 'edited_df' from data_editor (if hide_index=True) might reset index?
                # documentation says: "The index is preserved".

                # So we can update 'full_df' using the index from 'edited_df'
                st.session_state[file_key].update(edited_df)

                # Rerun to refresh the 'dirty' state
                st.rerun()
            else:
                st.warning(
                    "All changed rows were filtered out (irrelevant or empty category)."
                )

        except Exception as e:
            st.error(f"Error saving: {e}")


if __name__ == "__main__":
    main()
