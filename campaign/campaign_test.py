from campaign.Campaign import Campaign
import json

with open("campaign/email_addresses.json", "r") as f:
    email_addresses = json.load(f)

campaign = Campaign(email_addresses=email_addresses)


def test_1():  # generate_templates_from_context_list
    with open("campaign/contexts.json", "r") as f:
        contexts = json.load(f)

        context_list_temp = [
            (
                contexts[category],
                [
                    category,
                ],
            )
            for category in contexts
        ]

        context_list = list()
        for contexts, targets in context_list_temp:
            for context in contexts:
                context_list.append((context, targets))

        campaign.generate_templates_from_context_list(context_list=context_list)


def test_2():  # send emails
    campaign.carry_out_campaign()


test_1()
