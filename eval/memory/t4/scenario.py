"""The T4 scenario: a hand-written 10-day story of the user, Jamie Park, plus a pool of distractors.

`gen.py` turns this into fixtures: it assigns record ids, picks and times the distractors with the seed,
and writes days/, facts.yaml and probes.yaml. Everything here is synthetic: the people, the companies, the
addresses and the `.example` mail domains.

Records carry `resolved`: the values that a relative expression in the text stands for ("this Saturday"
-> date:2026-04-11, "9 tonight" -> time:21:00). The oracle's `source_precision` adds them to what it
extracts from the record's text, since a good memory entry writes the absolute value.

Items are the gold memory store: what an ideal agent keeps, in versions by day. The scripted agent of the
self-test writes and updates them through pab, and the reader's ceiling run reads their final versions.
"""

from __future__ import annotations

from typing import Any

START = "2026-04-06"  # a Monday; the ten days stay inside US daylight time
DAYS = 10
TZ = "-04:00"
OWNER = "jamie_park"
OWNER_MAIL = "Jamie Park <jamie.park@mailbox.example>"
OWNER_WORK_MAIL = "Jamie Park <jamie.park@northwind.example>"

# id -> (type, canonical name, aliases). The reader's answers and the oracle's value extraction both
# resolve names through these aliases; the owner's own aliases are never a key value.
ENTITIES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "jamie_park": ("person", "Jamie Park", ("Jamie",)),
    "maya_lindqvist": ("person", "Maya Lindqvist", ("Maya",)),
    "priya_raman": ("person", "Priya Raman", ("Priya",)),
    "dev_kapoor": ("person", "Dev Kapoor", ("Dev",)),
    "tom_whitfield": ("person", "Tom Whitfield", ("Tom",)),
    "elena_brooks": ("person", "Elena Brooks", ("Elena", "Lena", "Lena Brooks")),
    "omar_haddad": ("person", "Omar Haddad", ("Omar",)),
    "ana_sousa": ("person", "Ana Sousa", ("Dr. Sousa", "Dr. Ana Sousa", "Sousa")),
    "ken_ito": ("person", "Ken Ito", ("Dr. Ito", "Dr. Ken Ito", "Ito")),
    "lucia_ferrante": ("person", "Lucia Ferrante", ("Lucia",)),
    "kenji_mori": ("person", "Kenji Mori", ("Kenji",)),
    "grace_liu": ("person", "Grace Liu", ("Grace",)),
    "ben_okoro": ("person", "Ben Okoro", ("Ben",)),
    "hannah_reyes": ("person", "Hannah Reyes", ("Hannah",)),
    "imani_cole": ("person", "Imani Cole", ()),
    "sam_torres": ("person", "Sam Torres", ("Sam",)),
    "pine_street": ("place", "Pine Street Community Center", ("Pine Street",)),
    "harbor_hall": ("place", "Harbor Hall", ()),
    "harbor_room": ("place", "Harbor Room", ()),
    "lupa_trattoria": ("place", "Lupa Trattoria", ("Lupa",)),
    "kestrel_books": ("place", "Kestrel Books", ()),
    "paradise_lot": ("place", "Paradise Lot", ("The Paradise Lot",)),
    "studio_verde": ("place", "Studio Verde", ()),
    "room_4b": ("place", "Room 4B", ("4B",)),
    "room_7a": ("place", "Room 7A", ("7A",)),
    "denver": ("place", "Denver", ("DEN",)),
    "boston": ("place", "Boston", ("BOS",)),
    "riverside_dental": ("org", "Riverside Dental", ()),
    "adler_plumbing": ("org", "Adler Plumbing", ("Adler",)),
    "cedar_ridge": ("org", "Cedar Ridge Rentals", ("Cedar Ridge",)),
    "metro_auto": ("org", "Metro Auto", ()),
    "lumen_office": ("org", "Lumen Office", ()),
    "whitfield_properties": ("org", "Whitfield Properties", ()),
    "northwind": ("org", "Northwind", ()),
    "glass_orchard": ("title", "The Glass Orchard", ("Glass Orchard",)),
    "salt_and_signal": ("title", "Salt and Signal", ("Salt & Signal",)),
    "static_bloom": ("title", "Static Bloom", ()),
    "flight_ua1482": ("code", "UA 1482", ("UA1482", "United 1482")),
    "flight_ua990": ("code", "UA 990", ("UA990", "United 990")),
}

ADDRESSES = {
    "maya_lindqvist": "Maya Lindqvist <maya.lindqvist@northwind.example>",
    "priya_raman": "Priya Raman <priya.raman@mailbox.example>",
    "tom_whitfield": "Tom Whitfield <tom@whitfieldproperties.example>",
    "elena_brooks": "Elena Brooks <lena.brooks@mailbox.example>",
    "lucia_ferrante": "Lucia Ferrante <lucia.ferrante@mailbox.example>",
}


def mail(key: str, day: int, time: str, sender: str, subject: str, body: str, *,
         to: str = OWNER_MAIL, resolved: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"key": key, "day": day, "time": time, "kind": "mail", "sender": ADDRESSES.get(sender, sender),
            "to": to, "subject": subject, "body": body, "resolved": list(resolved)}


def msg(key: str, day: int, time: str, who: str, text: str, *, from_me: bool = False,
        resolved: tuple[str, ...] = ()) -> dict[str, Any]:
    name = ENTITIES[who][1] if who in ENTITIES else who
    return {"key": key, "day": day, "time": time, "kind": "message", "chat": name,
            "sender": "me" if from_me else name, "is_from_me": from_me, "text": text,
            "resolved": list(resolved)}


def event(key: str, day: int, time: str, title: str, start: str, end: str, *, location: str = "",
          notes: str = "", calendar: str = "Personal", recurrence: str | None = None,
          resolved: tuple[str, ...] = ()) -> dict[str, Any]:
    out = {"key": key, "day": day, "time": time, "kind": "event", "title": title, "start": start,
           "end": end, "all_day": False, "location": location, "notes": notes, "calendar": calendar,
           "resolved": list(resolved)}
    if recurrence:
        out["recurrence"] = recurrence
    return out


# fmt: off
RECORDS: list[dict[str, Any]] = [
    # Day 1, Monday 2026-04-06
    mail("A1", 1, "08:42", "maya_lindqvist", "Q2 planning offsite: save the date",
         "Hi team,\n\nOur Q2 planning offsite is on Thursday, April 16, from 9:30 AM to 4:00 PM at the Pine Street "
         "Community Center. Lunch will be catered by Lupa Trattoria. Please bring your team's Q2 draft.\n\n"
         "Thanks,\nMaya", to="Platform team <platform@northwind.example>"),
    mail("R1", 1, "10:05", "tom_whitfield", "Lease renewal for 14 Orchard Lane, Apt 3",
         "Hi Jamie,\n\nThanks for renewing. Starting next month your rent goes from $1,980 to $2,150 per month. "
         "Everything else in the lease stays the same.\n\nBest,\nTom Whitfield\nWhitfield Properties",
         resolved=("date:2026-05-01",)),
    mail("L0", 1, "13:30", "elena_brooks", "Game night?",
         "Hi Jamie! My partner Omar and I are finally hosting that game night at our place on Friday, April 24, "
         "around 7. Bring a snack if you can.\n\n- Lena (Elena Brooks)", resolved=("time:19:00",)),
    event("Y1", 1, "07:30", "Yoga", "2026-04-11T09:00:00-04:00", "2026-04-11T10:00:00-04:00",
          location="Studio Verde", notes="Instructor: Hannah Reyes. Weekly on Saturdays.",
          recurrence="FREQ=WEEKLY;BYDAY=SA"),
    msg("C1", 1, "19:12", "priya_raman",
        "Hey little bro!! Dev and I booked flights to come see you the weekend after next 🎉 Flight details "
        "soon. Love, your big sis", resolved=("date:2026-04-18", "date:2026-04-19")),

    # Day 2, Tuesday 2026-04-07
    mail("B1", 2, "09:20", "Riverside Dental <appointments@riversidedental.example>", "Appointment confirmation",
         "Dear Jamie Park,\n\nThis confirms your dental cleaning with Dr. Ana Sousa on Tuesday, April 21, 2026 at "
         "8:15 AM at Riverside Dental, 220 Water Street.\n\nRiverside Dental"),
    msg("K0", 2, "11:15", "kenji_mori", "booked UA 990 for the Seattle conference on May 4, finally"),
    mail("E1", 2, "12:02", "lucia_ferrante", "Book club: April pick",
         "Hi all,\n\nThis month we're reading The Glass Orchard by Imani Cole. We meet Wednesday the 22nd at "
         "7 PM at Kestrel Books.\n\nLucia", to="Book club <bookclub@lists.mailbox.example>",
         resolved=("date:2026-04-22",)),
    msg("F1", 2, "17:48", "omar_haddad",
        "Got 3 tickets for Static Bloom at the Paradise Lot this Saturday, doors at 8! $84.50 each. One's for "
        "you, one's for Lena. Venmo me for yours when you can", resolved=("date:2026-04-11", "time:20:00")),

    # Day 3, Wednesday 2026-04-08
    msg("D1", 3, "08:55", "tom_whitfield",
        "Hi Jamie, the plumber from Adler Plumbing will come this Friday between 1 and 3 PM to fix the kitchen "
        "sink. - Tom", resolved=("date:2026-04-10", "time:13:00")),
    event("A2", 3, "09:30", "Q2 planning offsite", "2026-04-16T09:30:00-04:00", "2026-04-16T16:00:00-04:00",
          location="Pine Street Community Center", notes="Organizer: Maya Lindqvist", calendar="Work"),
    event("H1", 3, "10:15", "Reading group: sparse attention survey", "2026-04-09T15:00:00-04:00",
          "2026-04-09T16:00:00-04:00", location="Room 4B", notes="Presenter: Grace Liu", calendar="Work"),
    mail("D2", 3, "11:40", "Adler Plumbing <office@adlerplumbing.example>", "Estimate for kitchen sink repair",
         "Hello,\n\nEstimate for the kitchen sink repair at 14 Orchard Lane, Apt 3: $180.00 (parts and labor). "
         "Technician: Ben Okoro.\n\nAdler Plumbing"),
    msg("T7", 3, "16:20", "kenji_mori", "Heads up: code freeze is the day after tomorrow at 5 PM.",
        resolved=("date:2026-04-10",)),
    msg("Y2", 3, "18:30", "hannah_reyes",
        "Hi everyone! Starting this Saturday, our class is at 10 instead of 9. See you on the mat - Hannah",
        resolved=("date:2026-04-11", "time:10:00", "time:09:00")),

    # Day 4, Thursday 2026-04-09
    mail("C2", 4, "07:58", "priya_raman", "Fwd: Your United itinerary",
         "Here it is! - your big sis\n\n---------- Forwarded message ----------\nFlight UA 1482\nDenver (DEN) to "
         "Boston (BOS)\nArrives Friday, April 17 at 6:40 PM\nPassengers: Priya Raman, Dev Kapoor"),
    msg("H2", 4, "13:05", "grace_liu", "Reading group is in Room 7A today, 4B got double-booked.",
        resolved=("date:2026-04-09",)),
    mail("G1", 4, "15:30", "Cedar Ridge Rentals <bookings@cedarridge.example>", "Booking request: Cabin #3",
         "Hi Jamie,\n\nWe received your booking request for Cedar Ridge Cabin #3: check-in June 12, check-out "
         "June 15 (3 nights). A deposit of $250.00 is due by April 20 to hold the dates.\n\nCedar Ridge Rentals"),
    msg("R2", 4, "18:10", "tom_whitfield",
        "Correction on the lease: the new rent will be $2,095, not $2,150. I misread the schedule, sorry!",
        resolved=("date:2026-05-01",)),

    # Day 5, Friday 2026-04-10
    msg("D3", 5, "09:10", "tom_whitfield",
        "Ben from Adler says the part is on backorder, so he'll come Monday instead, same 1-3 window.",
        resolved=("date:2026-04-13", "time:13:00", "time:15:00")),
    msg("E2", 5, "10:45", "lucia_ferrante",
        "Change of plan: we're reading Salt and Signal instead. The library copies of The Glass Orchard fell "
        "through."),
    msg("F2", 5, "12:30", "omar_haddad", "Sent you the $84.50 for my ticket yesterday, did it go through?",
        from_me=True, resolved=("date:2026-04-09",)),
    msg("L1", 5, "12:41", "omar_haddad",
        "Yep got it, thanks! Also: Lena's birthday is next Wednesday. I booked Harbor Room at 7, want to join? "
        "Don't tell her", resolved=("date:2026-04-15", "time:19:00")),
    mail("V1", 5, "14:00", "Metro Auto <service@metroauto.example>", "Service appointment confirmed",
         "Hi Jamie,\n\nYour 2019 Honda Civic is booked for service on Saturday, April 25 at 9:00 AM. "
         "Estimate: $129.99.\n\nMetro Auto"),
    mail("X5", 5, "16:20", "Harbor Hall Events <events@harborhall.example>", "Spring open house",
         "Planning a wedding or a company event? Tour Harbor Hall at our spring open house on Sunday, May 3, "
         "from noon to 4 PM."),

    # Day 6, Saturday 2026-04-11
    mail("A3", 6, "09:05", "maya_lindqvist", "Offsite moved",
         "Hi team,\n\nChange of plans: Pine Street had a water leak, so the Q2 planning offsite moves to Friday, "
         "April 17, same hours, at Harbor Hall (2nd floor).\n\nMaya",
         to="Platform team <platform@northwind.example>", resolved=("time:09:30", "time:16:00")),
    mail("X6", 6, "10:00", "Kestrel Books <events@kestrelbooks.example>", "Author event",
         "Imani Cole reads from The Glass Orchard at Kestrel Books on Thursday, April 30 at 6:30 PM."),
    mail("P1", 6, "11:25", "Lumen Office <orders@lumenoffice.example>", "Order confirmed",
         "Thanks for your order! Item: Ergo standing desk (walnut). Estimated delivery: Tuesday.",
         resolved=("date:2026-04-14",)),
    msg("F3", 6, "16:50", "elena_brooks", "Doors pushed to 9 tonight, the opener cancelled!",
        resolved=("date:2026-04-11", "time:21:00")),
    mail("X7", 6, "17:30", "Live Events <tickets@liveevents.example>", "Static Bloom adds a second show",
         "Static Bloom adds a second show at the Paradise Lot on Sunday, April 12! Tickets from $79.00."),

    # Day 7, Sunday 2026-04-12
    mail("G2", 7, "10:12", "Cedar Ridge Rentals <bookings@cedarridge.example>", "Deposit received",
         "Hi Jamie,\n\nWe received your deposit for Cabin #3 (June 12 to June 15). The remaining balance of "
         "$615.00 is due June 1.\n\nCedar Ridge Rentals"),
    msg("C3", 7, "14:33", "dev_kapoor",
        "Hey Jamie, Dev here. United rebooked us, we now land at 8:55 PM Friday. Same flight number.",
        resolved=("date:2026-04-17",)),
    msg("M1", 7, "18:02", "maya_lindqvist", "Can we do a quick 1:1 tomorrow at 11?",
        resolved=("date:2026-04-13", "time:11:00")),

    # Day 8, Monday 2026-04-13
    mail("B2", 8, "08:30", "Riverside Dental <appointments@riversidedental.example>", "Your appointment has changed",
         "Dear Jamie Park,\n\nDr. Sousa is unavailable on April 21. Your cleaning has been moved to Thursday, "
         "April 23 at 10:45 AM with Dr. Ken Ito.\n\nRiverside Dental"),
    mail("H3", 8, "09:00", "Northwind HR <hr@northwind.example>", "Benefits open enrollment",
         "Open enrollment for 2026-27 benefits closes two weeks from today. Make your selections in the portal.",
         to=OWNER_WORK_MAIL, resolved=("date:2026-04-27",)),
    msg("F4", 8, "12:15", "omar_haddad",
        "Lena says you're covering her ticket as her birthday present 😄 just send it whenever"),
    mail("D4", 8, "16:45", "Adler Plumbing <office@adlerplumbing.example>", "Invoice: kitchen sink repair",
         "Invoice for the kitchen sink repair at 14 Orchard Lane, Apt 3: $215.00 (the part price went up). Billed "
         "to Whitfield Properties.\n\nAdler Plumbing"),
    msg("C4", 8, "20:20", "priya_raman",
        "Booked us a table at Lupa Trattoria for Saturday at 7:30, the four of us with Omar 🍝", from_me=True,
        resolved=("date:2026-04-18", "time:19:30")),

    # Day 9, Tuesday 2026-04-14
    mail("E3", 9, "08:15", "lucia_ferrante", "Book club next week",
         "Reminder for next week's meeting at Kestrel Books: Grace Liu is bringing snacks and Omar will lead the "
         "discussion.\n\nLucia", to="Book club <bookclub@lists.mailbox.example>"),
    msg("A4", 9, "11:50", "kenji_mori",
        "Are you driving to Harbor Hall on Friday? Could I get a ride? I can meet you at the Elm St stop.",
        resolved=("date:2026-04-17",)),
    mail("P2", 9, "13:20", "Lumen Office <orders@lumenoffice.example>", "Delivery update",
         "Your Ergo standing desk is delayed and will now arrive Thursday.", resolved=("date:2026-04-16",)),
    mail("V2", 9, "15:05", "Metro Auto <service@metroauto.example>", "Revised estimate",
         "Revised estimate for your April 25 service: $164.99 (front brake pads added).\n\nMetro Auto"),

    # Day 10, Wednesday 2026-04-15
    msg("K2", 10, "10:30", "kenji_mori", "Can we do the PR review Thursday at 2 PM?",
        resolved=("date:2026-04-16",)),
    event("A5", 10, "11:00", "Q2 planning offsite", "2026-04-17T09:30:00-04:00", "2026-04-17T16:00:00-04:00",
          location="Harbor Hall, 2nd floor", notes="Organizer: Maya Lindqvist", calendar="Work"),
    msg("L2", 10, "17:15", "omar_haddad", "See you at Harbor Room at 7 tonight! 🎂",
        resolved=("date:2026-04-15", "time:19:00")),
    msg("C5", 10, "21:40", "priya_raman", "Can't wait to see you Friday!!", resolved=("date:2026-04-17",)),
]

# Distractors: gen.py picks DISTRACTORS_PER_DAY of them per day with the seed, without replacement, and times
# each inside its window. None of them answers a probe, abstention probes included.
DISTRACTORS_PER_DAY = 4
DISTRACTORS: list[dict[str, Any]] = [
    {"kind": "mail", "window": ("06:00", "09:00"), "sender": "Northwind Comms <comms@northwind.example>",
     "subject": "Northwind Weekly", "to": OWNER_WORK_MAIL,
     "body": "This week: a new espresso machine on the 3rd floor, the all-hands recording is posted, and the "
             "garage elevator is under maintenance."},
    {"kind": "mail", "window": ("07:00", "12:00"), "sender": "BeanBox <orders@beanbox.example>",
     "subject": "Your coffee subscription has shipped",
     "body": "Your monthly beans (Ethiopia Guji) are on the way. Total charged: $18.00."},
    {"kind": "mail", "window": ("08:00", "18:00"), "sender": "City Library <holds@citylibrary.example>",
     "subject": "Hold ready for pickup", "body": "Your hold 'The Overstory' is ready for pickup for seven days."},
    {"kind": "mail", "window": ("06:00", "10:00"), "sender": "Weather Alerts <alerts@weather.example>",
     "subject": "Wind advisory", "body": "A wind advisory is in effect until 8 PM today."},
    {"kind": "mail", "window": ("05:00", "08:00"), "sender": "Pods <feeds@pods.example>",
     "subject": "New episode: Systems Weekly", "body": "New episode of Systems Weekly: 'Caching at scale'."},
    {"kind": "mail", "window": ("12:00", "14:00"), "sender": "Corner Deli <receipts@cornerdeli.example>",
     "subject": "Your receipt", "body": "Turkey club and iced tea. Total: $13.75. Thanks for stopping by!"},
    {"kind": "mail", "window": ("09:00", "17:00"), "sender": "City Pharmacy <rx@citypharmacy.example>",
     "subject": "Your refill is ready", "body": "Your refill is ready for pickup at the Main St location."},
    {"kind": "mail", "window": ("09:00", "20:00"), "sender": "Riverside Run Club <runclub@mailbox.example>",
     "subject": "Sunday long run", "to": "Run club <runclub@lists.mailbox.example>",
     "body": "The Sunday long run starts at 7:30 AM from the boathouse: 10 miles at an easy pace."},
    {"kind": "mail", "window": ("09:00", "18:00"), "sender": "CodeHub <noreply@codehub.example>",
     "subject": "Review requested", "to": OWNER_WORK_MAIL,
     "body": "Kenji Mori requested your review on 'Refactor scheduler queue'."},
    {"kind": "mail", "window": ("10:00", "20:00"), "sender": "Streamly <hello@streamly.example>",
     "subject": "New season available", "body": "Season 3 of Harbor Lights is now streaming."},
    {"kind": "mail", "window": ("08:00", "20:00"), "sender": "SkyFares <deals@skyfares.example>",
     "subject": "Spring sale", "body": "Spring sale: round trips to Denver from $129. Book by Sunday."},
    {"kind": "mail", "window": ("08:00", "18:00"), "sender": "City Power <billing@citypower.example>",
     "subject": "Your bill is ready", "body": "Your April electricity bill of $64.20 is due May 3."},
    {"kind": "mail", "window": ("09:00", "17:00"), "sender": "Northwind IT <it@northwind.example>",
     "subject": "Laptop refresh", "to": OWNER_WORK_MAIL,
     "body": "Your laptop refresh is scheduled for May 12. Please back up your files before then."},
    {"kind": "mail", "window": ("06:00", "09:00"), "sender": "First Harbor Bank <alerts@firstharborbank.example>",
     "subject": "Statement available", "body": "Your March statement is available in online banking."},
    {"kind": "mail", "window": ("10:00", "19:00"), "sender": "ShipFast <tracking@shipfast.example>",
     "subject": "Delivered", "body": "Your package (USB-C cable, 2 m) was delivered to the front desk."},
    {"kind": "mail", "window": ("08:00", "20:00"), "sender": "Ironworks Gym <info@ironworksgym.example>",
     "subject": "New spin classes", "body": "New spin classes on Tuesdays at 6 PM. Bring a towel."},
    {"kind": "mail", "window": ("07:00", "21:00"), "sender": "Recipes Daily <recipes@cooking.example>",
     "subject": "Five weeknight pastas", "body": "Five weeknight pastas you can make in 20 minutes."},
    {"kind": "mail", "window": ("09:00", "17:00"), "sender": "Northwind Facilities <facilities@northwind.example>",
     "subject": "Badge readers", "to": OWNER_WORK_MAIL,
     "body": "Badge readers on floors 2-4 will be updated overnight. Expect a short outage."},
    {"kind": "mail", "window": ("08:00", "20:00"), "sender": "Green Thumb Nursery <hello@greenthumb.example>",
     "subject": "Seedlings are in", "body": "Tomato and basil seedlings are in stock this week."},
    {"kind": "mail", "window": ("08:00", "20:00"), "sender": "Museum of Science <members@mos.example>",
     "subject": "Members' night", "body": "Members' night returns next month with the new planetarium show."},
    {"kind": "message", "window": ("09:00", "18:00"), "who": "kenji_mori", "text": "lol did you see the all-hands slides"},
    {"kind": "message", "window": ("12:00", "22:00"), "who": "omar_haddad", "text": "bring earplugs haha"},
    {"kind": "message", "window": ("09:00", "21:00"), "who": "sam_torres",
     "text": "Hey, it's Sam from 3B. Your neighbor's cat got into the hallway again, no harm done"},
    {"kind": "message", "window": ("10:00", "18:00"), "who": "grace_liu", "text": "thanks for the feedback on my slides!"},
    {"kind": "message", "window": ("10:00", "21:00"), "who": "lucia_ferrante",
     "text": "Did anyone find my umbrella after the last meeting?"},
    {"kind": "message", "window": ("07:00", "20:00"), "who": "hannah_reyes",
     "text": "Mats are being cleaned this week, bring your own if you have one"},
    {"kind": "message", "window": ("09:00", "22:00"), "who": "priya_raman",
     "text": "Mom says hi and asks if you're eating vegetables"},
    {"kind": "message", "window": ("09:00", "22:00"), "who": "dev_kapoor", "text": "Did you ever finish that puzzle box?"},
    {"kind": "message", "window": ("12:00", "16:00"), "who": "kenji_mori", "text": "coffee?"},
    {"kind": "message", "window": ("10:00", "22:00"), "who": "elena_brooks", "text": "Omar is terrible at keeping secrets fyi"},
    {"kind": "message", "window": ("09:00", "18:00"), "who": "maya_lindqvist", "text": "Thanks for the update on the migration"},
    {"kind": "message", "window": ("08:00", "20:00"), "who": "tom_whitfield",
     "text": "Reminder: the building's fire alarm test is this week, sorry in advance for the noise"},
    {"kind": "message", "window": ("17:00", "22:00"), "who": "omar_haddad", "text": "did you watch the game??"},
    {"kind": "message", "window": ("08:00", "11:00"), "who": "maya_lindqvist", "text": "running 5 late to standup"},
    {"kind": "message", "window": ("19:00", "23:00"), "who": "priya_raman", "text": "look at this dog 😂"},
    {"kind": "event", "window": ("06:00", "08:00"), "title": "Team standup", "calendar": "Work", "at": ("09:45", "10:00")},
    {"kind": "event", "window": ("06:00", "08:00"), "title": "Focus time", "calendar": "Work", "at": ("13:00", "15:00")},
    {"kind": "event", "window": ("06:00", "08:00"), "title": "Laundry", "calendar": "Personal", "at": ("19:00", "20:00")},
    {"kind": "event", "window": ("06:00", "08:00"), "title": "Groceries", "calendar": "Personal", "at": ("18:00", "18:45")},
    {"kind": "event", "window": ("06:00", "08:00"), "title": "Water the plants", "calendar": "Personal", "at": ("08:00", "08:15")},
]

# The gold memory store. A version's sources are record keys; its text uses absolute dates and only values
# found in (or resolved for) those records, so the gold store has source_precision 1.
ITEMS: list[dict[str, Any]] = [
    {"key": "offsite", "versions": [
        {"day": 1, "sources": ["A1"], "text": "Q2 planning offsite (work; organizer Maya Lindqvist): Thursday "
         "2026-04-16, 09:30-16:00, at Pine Street Community Center. Lunch catered by Lupa Trattoria. Bring the "
         "team's Q2 draft."},
        {"day": 6, "sources": ["A1", "A3"], "text": "Q2 planning offsite (work; organizer Maya Lindqvist): moved "
         "to Friday 2026-04-17, 09:30-16:00, at Harbor Hall (2nd floor), because Pine Street had a water leak. "
         "Lunch catered by Lupa Trattoria. Bring the team's Q2 draft."},
    ]},
    {"key": "rent", "versions": [
        {"day": 1, "sources": ["R1"], "text": "Lease at 14 Orchard Lane, Apt 3, landlord Tom Whitfield (Whitfield "
         "Properties): rent goes from $1,980 to $2,150 per month starting 2026-05-01."},
        {"day": 4, "sources": ["R1", "R2"], "text": "Lease at 14 Orchard Lane, Apt 3, landlord Tom Whitfield "
         "(Whitfield Properties): new rent $2,095 per month (corrected from $2,150) starting 2026-05-01; the old "
         "rent is $1,980."},
    ]},
    {"key": "lena", "versions": [
        {"day": 1, "sources": ["L0"], "text": "Elena (Lena) Brooks is Omar Haddad's partner. They host a game "
         "night at their place on Friday 2026-04-24 around 19:00."},
    ]},
    {"key": "yoga", "versions": [
        {"day": 1, "sources": ["Y1"], "text": "Yoga at Studio Verde with instructor Hannah Reyes, weekly on "
         "Saturdays, 09:00-10:00."},
        {"day": 3, "sources": ["Y1", "Y2"], "text": "Yoga at Studio Verde with instructor Hannah Reyes, weekly on "
         "Saturdays: from Saturday 2026-04-11 the class is at 10:00 instead of 09:00."},
    ]},
    {"key": "sister", "versions": [
        {"day": 1, "sources": ["C1"], "text": "Priya Raman is Jamie's older sister. Priya and Dev are visiting "
         "the weekend of 2026-04-18 to 2026-04-19."},
    ]},
    {"key": "dentist", "versions": [
        {"day": 2, "sources": ["B1"], "text": "Dental cleaning at Riverside Dental (220 Water Street) with "
         "Dr. Ana Sousa: Tuesday 2026-04-21 at 08:15."},
        {"day": 8, "sources": ["B1", "B2"], "text": "Dental cleaning at Riverside Dental (220 Water Street): "
         "moved to Thursday 2026-04-23 at 10:45 with Dr. Ken Ito, because Dr. Sousa is unavailable on "
         "2026-04-21 (originally 08:15 with Dr. Ana Sousa)."},
    ]},
    {"key": "kenji_trip", "versions": [
        {"day": 2, "sources": ["K0"], "text": "Kenji Mori flies UA 990 to a conference in Seattle on 2026-05-04."},
    ]},
    {"key": "book_club", "versions": [
        {"day": 2, "sources": ["E1"], "text": "Book club (organizer Lucia Ferrante): reading The Glass Orchard by "
         "Imani Cole; meeting Wednesday 2026-04-22 at 19:00 at Kestrel Books."},
        {"day": 5, "sources": ["E1", "E2"], "text": "Book club (organizer Lucia Ferrante): now reading Salt and "
         "Signal instead of The Glass Orchard; meeting Wednesday 2026-04-22 at 19:00 at Kestrel Books."},
        {"day": 9, "sources": ["E1", "E2", "E3"], "text": "Book club (organizer Lucia Ferrante): reading Salt and "
         "Signal; meeting Wednesday 2026-04-22 at 19:00 at Kestrel Books. Grace Liu brings snacks; Omar leads "
         "the discussion."},
    ]},
    {"key": "concert", "versions": [
        {"day": 2, "sources": ["F1"], "text": "Static Bloom concert at the Paradise Lot on Saturday 2026-04-11, "
         "doors at 20:00. Omar bought 3 tickets at $84.50 each (Jamie, Omar, Lena); Jamie owes Omar $84.50."},
        {"day": 5, "sources": ["F1", "F2"], "text": "Static Bloom concert at the Paradise Lot on Saturday "
         "2026-04-11, doors at 20:00. Omar bought 3 tickets at $84.50 each (Jamie, Omar, Lena). Jamie paid Omar "
         "$84.50 for his own ticket on 2026-04-09."},
        {"day": 6, "sources": ["F1", "F2", "F3"], "text": "Static Bloom concert at the Paradise Lot on Saturday "
         "2026-04-11: doors moved to 21:00 (the opener cancelled; Lena told Jamie). Omar bought 3 tickets at "
         "$84.50 each (Jamie, Omar, Lena). Jamie paid Omar $84.50 for his own ticket on 2026-04-09."},
        {"day": 8, "sources": ["F1", "F2", "F3", "F4"], "text": "Static Bloom concert at the Paradise Lot on "
         "Saturday 2026-04-11, doors at 21:00. Omar bought 3 tickets at $84.50 each (Jamie, Omar, Lena). Jamie "
         "paid Omar $84.50 for his own ticket on 2026-04-09, and is also covering Lena's ticket as her birthday "
         "present, which he still has to pay Omar for."},
    ]},
    {"key": "plumber", "versions": [
        {"day": 3, "sources": ["D1", "D2"], "text": "Kitchen sink repair at 14 Orchard Lane, Apt 3, arranged by "
         "landlord Tom: Adler Plumbing, technician Ben Okoro, Friday 2026-04-10 between 13:00 and 15:00. "
         "Estimate $180.00."},
        {"day": 5, "sources": ["D1", "D2", "D3"], "text": "Kitchen sink repair at 14 Orchard Lane, Apt 3, "
         "arranged by landlord Tom: Adler Plumbing, technician Ben Okoro. The part is on backorder, so the visit "
         "moved from Friday 2026-04-10 to Monday 2026-04-13, 13:00-15:00. Estimate $180.00."},
        {"day": 8, "sources": ["D1", "D2", "D3", "D4"], "text": "Kitchen sink repair at 14 Orchard Lane, Apt 3, "
         "arranged by landlord Tom: Adler Plumbing, technician Ben Okoro, visit Monday 2026-04-13 (moved from "
         "2026-04-10, part on backorder). Final invoice $215.00 (estimate was $180.00), billed to Whitfield "
         "Properties."},
    ]},
    {"key": "reading_group", "versions": [
        {"day": 3, "sources": ["H1"], "text": "Reading group: Grace Liu presents a sparse attention survey on "
         "Thursday 2026-04-09, 15:00-16:00, in Room 4B."},
        {"day": 4, "sources": ["H1", "H2"], "text": "Reading group: Grace Liu presents a sparse attention survey "
         "on Thursday 2026-04-09, 15:00-16:00; moved to Room 7A because 4B was double-booked."},
    ]},
    {"key": "code_freeze", "versions": [
        {"day": 3, "sources": ["T7"], "text": "Code freeze on Friday 2026-04-10 at 17:00 (from Kenji Mori)."},
    ]},
    {"key": "flight", "versions": [
        {"day": 4, "sources": ["C2"], "text": "Priya Raman and Dev Kapoor fly UA 1482 from Denver (DEN) to Boston "
         "(BOS), arriving Friday 2026-04-17 at 18:40."},
        {"day": 7, "sources": ["C2", "C3"], "text": "Priya Raman and Dev Kapoor fly UA 1482 from Denver (DEN) to "
         "Boston (BOS) on Friday 2026-04-17. United rebooked them (Dev told Jamie): they now arrive at 20:55 "
         "instead of 18:40, same flight number."},
    ]},
    {"key": "cabin", "versions": [
        {"day": 4, "sources": ["G1"], "text": "Cedar Ridge Cabin #3 (Cedar Ridge Rentals): check-in 2026-06-12, "
         "check-out 2026-06-15 (3 nights). Deposit $250.00 due by 2026-04-20."},
        {"day": 7, "sources": ["G1", "G2"], "text": "Cedar Ridge Cabin #3 (Cedar Ridge Rentals): check-in "
         "2026-06-12, check-out 2026-06-15 (3 nights). Deposit of $250.00 received; remaining balance $615.00 "
         "due 2026-06-01."},
    ]},
    {"key": "lena_birthday", "versions": [
        {"day": 5, "sources": ["L1"], "text": "Lena's birthday dinner (a surprise; don't tell her): Wednesday "
         "2026-04-15 at 19:00 at Harbor Room, booked by Omar."},
    ]},
    {"key": "car", "versions": [
        {"day": 5, "sources": ["V1"], "text": "Metro Auto service for the 2019 Honda Civic: Saturday 2026-04-25 "
         "at 09:00. Estimate $129.99."},
        {"day": 9, "sources": ["V1", "V2"], "text": "Metro Auto service for the 2019 Honda Civic: Saturday "
         "2026-04-25 at 09:00. Revised estimate $164.99 (front brake pads added; was $129.99)."},
    ]},
    {"key": "desk", "versions": [
        {"day": 6, "sources": ["P1"], "text": "Ordered an Ergo standing desk (walnut) from Lumen Office; delivery "
         "expected Tuesday 2026-04-14."},
        {"day": 9, "sources": ["P1", "P2"], "text": "Ordered an Ergo standing desk (walnut) from Lumen Office; "
         "delivery delayed from 2026-04-14 to Thursday 2026-04-16."},
    ]},
    {"key": "one_on_one", "versions": [
        {"day": 7, "sources": ["M1"], "text": "1:1 with Maya Lindqvist on Monday 2026-04-13 at 11:00."},
    ]},
    {"key": "benefits", "versions": [
        {"day": 8, "sources": ["H3"], "text": "Northwind benefits open enrollment closes on 2026-04-27."},
    ]},
    {"key": "family_dinner", "versions": [
        {"day": 8, "sources": ["C1", "C4"], "text": "Dinner at Lupa Trattoria on Saturday 2026-04-18 at 19:30 with "
         "Priya, Dev and Omar (Jamie booked it)."},
    ]},
    {"key": "ride", "versions": [
        {"day": 9, "sources": ["A4"], "text": "Kenji Mori asked Jamie for a ride to Harbor Hall on Friday "
         "2026-04-17; he can meet at the Elm St stop."},
    ]},
    {"key": "pr_review", "versions": [
        {"day": 10, "sources": ["K2"], "text": "PR review with Kenji Mori on Thursday 2026-04-16 at 14:00."},
    ]},
]

# Answer types: date (YYYY-MM-DD), datetime (local YYYY-MM-DDTHH:MM), time (HH:MM), money (integer cents),
# entity (an ENTITIES id), list (a set of ENTITIES ids). Abstention probes expect null and keep a type for
# the reader's format hint. `support` lists the record keys a correct answer needs.
PROBES: list[dict[str, Any]] = [
    {"id": "ex01", "category": "extraction", "type": "time", "answer": "09:30", "support": ["A1"],
     "question": "At what time does the Q2 planning offsite start?"},
    {"id": "ex02", "category": "extraction", "type": "entity", "answer": "denver", "support": ["C2"],
     "question": "From which city does Priya and Dev's flight depart?"},
    {"id": "ex03", "category": "extraction", "type": "entity", "answer": "ben_okoro", "support": ["D2"],
     "question": "What is the name of the Adler Plumbing technician for Jamie's kitchen sink?"},
    {"id": "ex04", "category": "extraction", "type": "entity", "answer": "omar_haddad", "support": ["E3"],
     "question": "Who will lead the discussion at the book club meeting?"},
    {"id": "ex05", "category": "extraction", "type": "money", "answer": 8450, "support": ["F1"],
     "question": "How much did each Static Bloom ticket cost?"},
    {"id": "ex06", "category": "extraction", "type": "date", "answer": "2026-06-12", "support": ["G1"],
     "question": "What is the check-in date for Cedar Ridge Cabin #3?"},
    {"id": "ex07", "category": "extraction", "type": "money", "answer": 61500, "support": ["G2"],
     "question": "What is the remaining balance for the Cedar Ridge cabin?"},
    {"id": "ex08", "category": "extraction", "type": "date", "answer": "2026-06-01", "support": ["G2"],
     "question": "By what date is the remaining balance for the Cedar Ridge cabin due?"},
    {"id": "ex09", "category": "extraction", "type": "entity", "answer": "grace_liu", "support": ["H1"],
     "question": "Who presented at the reading group on April 9?"},
    {"id": "ex10", "category": "extraction", "type": "datetime", "answer": "2026-04-25T09:00", "support": ["V1"],
     "question": "When is Jamie's car booked for service at Metro Auto?"},
    {"id": "ex11", "category": "extraction", "type": "entity", "answer": "ana_sousa", "support": ["B1"],
     "question": "Which dentist was Jamie's cleaning originally booked with?"},
    {"id": "ex12", "category": "extraction", "type": "entity", "answer": "lumen_office", "support": ["P1"],
     "question": "From which company did Jamie order the standing desk?"},

    {"id": "cs01", "category": "cross_session", "type": "datetime", "answer": "2026-04-17T20:55",
     "support": ["C1", "C2", "C3"], "question": "When does the flight of Jamie's sister land?"},
    {"id": "cs02", "category": "cross_session", "type": "entity", "answer": "harbor_room", "support": ["L0", "L1"],
     "question": "At which restaurant is the birthday dinner for Omar's partner?"},
    {"id": "cs03", "category": "cross_session", "type": "money", "answer": 86500, "support": ["G1", "G2"],
     "question": "What is the total price of the Cedar Ridge cabin stay, deposit plus remaining balance?"},
    {"id": "cs04", "category": "cross_session", "type": "entity", "answer": "kenji_mori", "support": ["A3", "A4"],
     "question": "Which colleague asked Jamie for a ride to the Q2 planning offsite?"},
    {"id": "cs05", "category": "cross_session", "type": "date", "answer": "2026-04-22", "support": ["E1", "E3"],
     "question": "On what date will Omar lead a book club discussion?"},
    {"id": "cs06", "category": "cross_session", "type": "money", "answer": 8450, "support": ["F1", "F2", "F4"],
     "question": "How much does Jamie still owe Omar for Static Bloom tickets?"},
    {"id": "cs07", "category": "cross_session", "type": "entity", "answer": "dev_kapoor",
     "support": ["C1", "C2", "C3"], "question": "Who told Jamie that the flight of Jamie's sister had been rebooked?"},
    {"id": "cs08", "category": "cross_session", "type": "list", "answer": ["priya_raman", "dev_kapoor"],
     "support": ["A3", "C2"], "question": "Who arrives in Boston on the day of the rescheduled Q2 planning offsite?"},
    {"id": "cs09", "category": "cross_session", "type": "entity", "answer": "flight_ua1482", "support": ["C2", "C3"],
     "question": "What is the flight number of the rebooked flight that Dev mentioned?"},
    {"id": "cs10", "category": "cross_session", "type": "entity", "answer": "grace_liu", "support": ["H1", "E3"],
     "question": "Which person presented at the April 9 reading group and is also bringing snacks to the book club?"},
    {"id": "cs11", "category": "cross_session", "type": "entity", "answer": "lupa_trattoria", "support": ["A1", "A3"],
     "question": "Which restaurant is catering lunch at the event at Harbor Hall on April 17?"},
    {"id": "cs12", "category": "cross_session", "type": "entity", "answer": "studio_verde", "support": ["Y1", "Y2"],
     "question": "At which studio is the yoga class that moved to 10 AM?"},

    {"id": "tm01", "category": "temporal", "type": "date", "answer": "2026-04-11", "support": ["F1"],
     "question": "On what date is the Static Bloom concert that Omar bought tickets for?"},
    {"id": "tm02", "category": "temporal", "type": "date", "answer": "2026-04-10", "support": ["D1"],
     "question": "On what date was the plumber originally scheduled to come?"},
    {"id": "tm03", "category": "temporal", "type": "datetime", "answer": "2026-04-15T19:00", "support": ["L1"],
     "question": "When is Lena's birthday dinner?"},
    {"id": "tm04", "category": "temporal", "type": "date", "answer": "2026-04-14", "support": ["P1"],
     "question": "On what date was the standing desk originally expected to arrive?"},
    {"id": "tm05", "category": "temporal", "type": "datetime", "answer": "2026-04-13T11:00", "support": ["M1"],
     "question": "When is Jamie's 1:1 with Maya?"},
    {"id": "tm06", "category": "temporal", "type": "date", "answer": "2026-04-27", "support": ["H3"],
     "question": "On what date does Northwind's benefits open enrollment close?"},
    {"id": "tm07", "category": "temporal", "type": "datetime", "answer": "2026-04-10T17:00", "support": ["T7"],
     "question": "When is the code freeze?"},
    {"id": "tm08", "category": "temporal", "type": "date", "answer": "2026-04-09", "support": ["F2"],
     "question": "On what date did Jamie send Omar the money for Jamie's own concert ticket?"},
    {"id": "tm09", "category": "temporal", "type": "datetime", "answer": "2026-04-11T10:00", "support": ["Y2"],
     "question": "When is the first yoga class at the new 10 AM time?"},
    {"id": "tm10", "category": "temporal", "type": "date", "answer": "2026-05-01", "support": ["R1"],
     "question": "On what date does Jamie's new rent take effect?"},
    {"id": "tm11", "category": "temporal", "type": "datetime", "answer": "2026-04-18T19:30", "support": ["C4"],
     "question": "When is the dinner reservation at Lupa Trattoria with Priya and Dev?"},
    {"id": "tm12", "category": "temporal", "type": "datetime", "answer": "2026-04-16T14:00", "support": ["K2"],
     "question": "When is the PR review with Kenji?"},

    {"id": "ku01", "category": "knowledge_update", "type": "date", "answer": "2026-04-17", "support": ["A3"],
     "question": "On what date is the Q2 planning offsite?"},
    {"id": "ku02", "category": "knowledge_update", "type": "entity", "answer": "harbor_hall", "support": ["A3"],
     "question": "Where is the Q2 planning offsite held?"},
    {"id": "ku03", "category": "knowledge_update", "type": "datetime", "answer": "2026-04-23T10:45", "support": ["B2"],
     "question": "When is Jamie's dental cleaning?"},
    {"id": "ku04", "category": "knowledge_update", "type": "entity", "answer": "ken_ito", "support": ["B2"],
     "question": "Which dentist will do Jamie's dental cleaning?"},
    {"id": "ku05", "category": "knowledge_update", "type": "date", "answer": "2026-04-13", "support": ["D3"],
     "question": "On what date will the plumber come to fix the kitchen sink?"},
    {"id": "ku06", "category": "knowledge_update", "type": "money", "answer": 21500, "support": ["D4"],
     "question": "What is the final cost of the kitchen sink repair?"},
    {"id": "ku07", "category": "knowledge_update", "type": "entity", "answer": "salt_and_signal", "support": ["E2"],
     "question": "Which book is the book club reading this month?"},
    {"id": "ku08", "category": "knowledge_update", "type": "time", "answer": "21:00", "support": ["F3"],
     "question": "At what time do doors open for the Static Bloom show that Jamie is going to?"},
    {"id": "ku09", "category": "knowledge_update", "type": "entity", "answer": "room_7a", "support": ["H2"],
     "question": "In which room was the April 9 reading group held?"},
    {"id": "ku10", "category": "knowledge_update", "type": "date", "answer": "2026-04-16", "support": ["P2"],
     "question": "On what date will the standing desk arrive?"},
    {"id": "ku11", "category": "knowledge_update", "type": "money", "answer": 16499, "support": ["V2"],
     "question": "What is the current estimate for Jamie's car service at Metro Auto?"},
    {"id": "ku12", "category": "knowledge_update", "type": "money", "answer": 209500, "support": ["R2"],
     "question": "What will Jamie's new monthly rent be?"},

    {"id": "ab01", "category": "abstention", "type": "money", "answer": None, "support": [],
     "question": "What is the parking fee at the Q2 planning offsite venue?"},
    {"id": "ab02", "category": "abstention", "type": "entity", "answer": None, "support": [],
     "question": "Which part was on backorder for the kitchen sink repair?"},
    {"id": "ab03", "category": "abstention", "type": "entity", "answer": None, "support": [],
     "question": "Who is the author of Salt and Signal?"},
    {"id": "ab04", "category": "abstention", "type": "entity", "answer": None, "support": [],
     "question": "What is the door code for Cedar Ridge Cabin #3?"},
    {"id": "ab05", "category": "abstention", "type": "date", "answer": None, "support": [],
     "question": "On what date do Priya and Dev fly back home?"},
    {"id": "ab06", "category": "abstention", "type": "time", "answer": None, "support": [],
     "question": "What is the check-in time at Cedar Ridge Cabin #3?"},
    {"id": "ab07", "category": "abstention", "type": "entity", "answer": None, "support": [],
     "question": "How old is Lena turning on her birthday?"},
    {"id": "ab08", "category": "abstention", "type": "entity", "answer": None, "support": [],
     "question": "What is the order number of Jamie's standing desk order?"},
    {"id": "ab09", "category": "abstention", "type": "money", "answer": None, "support": [],
     "question": "How much did the birthday dinner at Harbor Room cost?"},
    {"id": "ab10", "category": "abstention", "type": "entity", "answer": None, "support": [],
     "question": "What is Kenji's home address?"},
    {"id": "ab11", "category": "abstention", "type": "entity", "answer": None, "support": [],
     "question": "In which room at Harbor Hall is the Q2 planning offsite held?"},
    {"id": "ab12", "category": "abstention", "type": "entity", "answer": None, "support": [],
     "question": "Who will present at the reading group on April 23?"},
]
# fmt: on
