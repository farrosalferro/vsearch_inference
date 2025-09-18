TEMPLATE = '''Task: Extract the object(s) being asked about in the user’s question.

Rules:
- Output only the object(s).
- If multiple objects, separate them with commas.
- Do not include verbs, adjectives, or locations.
- Always use lowercase.
- If no clear object is present, return "none".

Examples:

User: What is/are the object(s) that being asked in below question? If there are multiple objects, use comma to separate them.
'What kind of drink can we buy from that vending machine?'
Assistant: vending machine

User: What is/are the object(s) that being asked in below question? If there are multiple objects, use comma to separate them.
'Is the wallet on the left or right side of the keyboard?'
Assistant: wallet, keyboard

User: What is/are the object(s) that being asked in below question? If there are multiple objects, use comma to separate them.
'Did you bring the laptop and charger?'
Assistant: laptop, charger

User: What is/are the object(s) that being asked in below question? If there are multiple objects, use comma to separate them.
'Which button on the remote controls the TV volume?'
Assistant: button, remote, TV

User: What is/are the object(s) that being asked in below question? If there are multiple objects, use comma to separate them.
'Is it raining?'
Assistant: none

---

User: What is/are the object(s) that being asked in below question? If there are multiple objects, use comma to separate them.
'{question}'
'''