import pandas as pd
from inference import inference_pipeline
import os


def test_inference():
    # Create sample data
    data = {
        "Description": [
            "This is a short desc.",  # Should fail length check (>50 chars required)
            "Thïs hås nøn-åscii chåråcters so it should fail the English check.",  # Should fail ascii check
            "This is a perfectly valid description that is long enough to pass the quality check but should be irrelevant I hope. This is just a complaint about service.",  # Irrelevant
            "I received an email from a prince in Nigeria asking for my bank details to transfer $10 million USD. He needs a small fee first.",  # Scam
            "My computer has a virus and tech support called me to pay $500 in gift cards to fix it immediately.",  # Scam
        ]
    }

    df = pd.DataFrame(data)

    print("Input DataFrame:")
    print(df)
    print("\nRunning Inference Pipeline...")

    try:
        result_df = inference_pipeline(df)

        print("\nResults DataFrame:")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", None)
        print(result_df)

        # Basic Assertions
        assert len(result_df) == 5
        assert result_df.iloc[0]["Predicted Category"] == "Could not classify"
        assert result_df.iloc[1]["Predicted Category"] == "Could not classify"

        # Save to CSV for inspection
        result_df.to_csv("test_inference_results.csv", index=False)
        print("\nSaved results to test_inference_results.csv")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_inference()
