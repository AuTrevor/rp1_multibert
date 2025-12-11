import pandas as pd
import random
import csv

# Categories from taxonomy_mapping.csv
CATEGORIES = [
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

# Sample data templates
SCAM_TEMPLATES = {
    "Account or Identity Takeover Scams": [
        "Someone accessed my bank account without permission and transferred money.",
        "I received a notification that my password was changed but I didn't do it.",
        "My identity has been stolen and used to open credit cards.",
        "Hacker took over my social media account and is asking friends for money.",
    ],
    "Buying and Selling (Products or Services) Scams": [
        "I paid for a laptop on Marketplace but the seller blocked me and never sent it.",
        "The item I received was completely different from the description and photos.",
        "Seller demanded payment via gift cards only.",
        "I sold an item but the buyer sent a fake payment confirmation.",
    ],
    "Donation Scams": [
        "A fake charity website asked for donations for the earthquake victims.",
        "Someone pretending to be Red Cross called asking for credit card details.",
        "Crowdfunding campaign for a sick child turned out to be fake.",
    ],
    "Investment Scams": [
        "Detailed promise of 200% returns on crypto investment in one week.",
        "Binary options trading platform won't let me withdraw my funds.",
        "Ponzi scheme involving 'high yield' fictional forex trading.",
        "A broker contacted me on WhatsApp promising guaranteed profits.",
    ],
    "Jobs and Employment Scams": [
        "Job offer required me to pay for a 'starter kit' before working.",
        "I was hired for data entry but they sent a fake check for office supplies.",
        "Recruiter asked for bank details for 'verification' before interview.",
        "Work from home job processing payments involved money laundering.",
    ],
    "Money Recovery Scams": [
        "A company contacted me saying they can recover my lost crypto for a fee.",
        "Lawyer claimed he can get my scammed money back if I pay tax upfront.",
        "Recovery experts demanded access to my wallet to trace the funds.",
    ],
    "Payment Redirection/ Business Email Compromise Scams": [
        "Received an invoice that looked real but the bank details were changed.",
        "Email from CEO asking to urgently wire funds to a new supplier.",
        "Hacked vendor email sent a request to update payment information.",
    ],
    "Phishing Scams": [
        "Received text saying my delivery is on hold, click link to pay fee.",
        "Email claiming my Netflix subscription is expired, update card details.",
        "Fake tax office SMS saying I have a refund waiting.",
        "Bank alert saying unauthorized login, click here to secure account.",
    ],
    "Relationship Scams": [
        "Met someone online who needs money for a plane ticket to visit me.",
        "My online boyfriend needs money for emergency surgery.",
        "Romance scammer claims they are a soldier stuck overseas needing funds.",
        "Sent money to a person I met on a dating app who promised to marry me.",
    ],
    "Threat Scams": [
        "Received email claiming they have webcam footage of me, pay bitcoin or leak.",
        "Caller claiming to be police said I will be arrested if I don't pay fine.",
        "Tax office threat to seize assets unless immediate payment made via gift card.",
    ],
    "Unexpected Money Scams": [
        "Email saying I won a foreign lottery I never entered.",
        "Inheritance notification from a distant relative I've never heard of.",
        "Letter stating I have unclaimed funds but need to pay processing fee.",
    ],
    "Other Scams": [
        "Scam involving a fake rental property listing.",
        "Immigration visa scam promising guaranteed entry for a price.",
        "Fake tech support called saying my computer has a virus.",
    ],
}

IRRELEVANT_TEXTS = [
    "The application crashes every time I try to log in.",
    "I forgot my password and the reset link isn't working.",
    "Where can I find the nearest branch to my location?",
    "I want to close my account because the fees are too high.",
    "The user interface is very confusing and hard to navigate.",
    "My internet connection is slow today.",
    "Just testing the system.",
    "I need to update my mailing address.",
    "What are your opening hours on weekends?",
    "This is not a scam, I just have a complaint about service.",
    "Can I speak to a manager please?",
    "The coffee machine in the lobby is broken.",
    "sdgfsdfgsdfg junk text",
    "How do I apply for a credit card extension?",
]


def generate_data(num_rows=200):
    data = []

    for _ in range(num_rows):
        is_irrelevant = random.choice([0, 1])

        if is_irrelevant:
            # Irrelevant: irrelevant=1
            # We assign a random category just to fill the CSV column,
            # or maybe "Other Scams" or meaningful noise.
            # Often irrelevant reports might still be tagged with a category by users.
            description = (
                random.choice(IRRELEVANT_TEXTS) + " " + str(random.randint(1, 1000))
            )
            category = random.choice(CATEGORIES)
            irrelevant_flag = 1
        else:
            # Scam: irrelevant=0
            category = random.choice(CATEGORIES)
            # Pick a template and add some random variation to avoid duplicates
            base_text = random.choice(
                SCAM_TEMPLATES.get(category, SCAM_TEMPLATES["Other Scams"])
            )
            description = base_text + f" (Ref: {random.randint(1000, 9999)})"
            irrelevant_flag = 0

        data.append(
            {
                "Description": description,
                "Category": category,
                "irrelevant": irrelevant_flag,
            }
        )

    return data


if __name__ == "__main__":
    df = pd.DataFrame(generate_data(200))
    output_file = "functional_test_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} rows to {output_file}")
