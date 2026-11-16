"""In-game level titles for the level files in DUST/DustSave/Ingame.

The titles are not stored as readable text in the install (README), so they come from a walkthrough's
level list. Story maps: ALPHA_MapN = story level N in order (checked against live data for Map 2 "The
Ritual" and Map 11 "Emergence"; the rest follow the order, and the player saves MAP_nn = map nn+1 agree).
Challenge maps: matched by file name or meaning; five matched by elimination were confirmed in play, and
Drill by matching the live terrain, Gospel in play: all 30 confirmed. CHALLENGE_GUESSED is empty.
"""
import re

STORY = {1: 'The Breath', 2: 'The Ritual', 3: 'The Roar of the Sea', 4: 'The First Power',
         5: 'The Thousand Sources', 6: 'Wildfires', 7: 'Waves', 8: 'High Tides', 9: 'The Tears of Stone',
         10: 'The Raging Earth', 11: 'Emergence', 12: 'Movements', 13: 'Origins'}

# challenge file -> (walkthrough number C1..C30, title)
CHALLENGE = {
    'Defi_019_Fire': (1, 'Fire!'),
    'Defi_004_BailOut': (3, 'Bail Out'),
    'Defi_006_BigWave': (4, 'The Big Wave'),
    'Defi_007_Canioning': (5, 'Canyoning'),
    'Defi_027_PumpItUp': (6, 'Pump It Up'),
    'Defi_025_Molyneux': (7, 'Xuenylom'),              # "Molyneux" backwards
    'Defi_044_Aquasplash': (8, 'Aquasplash'),
    'Defi_031_Waterfall': (9, 'Waterfall'),
    'Defi_021_GuardianAngel': (10, 'Guardian Angel'),
    'Defi_017_Explosions': (11, 'Explosions'),
    'Defi_029_Sacrifice': (12, 'Sacrifice'),
    'Defi_024_Moise_2': (13, 'Gospel'),                # Moses (parting the water); confirmed in play
    'Defi_008_Collapse': (16, 'Collapse'),
    'Defi_015_EpicTsunamis': (17, 'Epic Tsunamis'),
    'Defi_011_Ditch': (18, 'Ditch'),
    'Defi_034_YouHave30Seconds': (19, 'Countdown'),
    'Defi_041_Migration': (20, 'Migration'),
    'Defi_038_Nobel': (21, 'Nitro'),                   # Nobel = dynamite
    'Defi_003_Atlantis': (23, 'Atlantis'),
    'Defi_022_Irony': (24, 'Irony'),
    'Defi_012_Dredge': (25, 'Drill'),                  # confirmed live (hard ground 100 %)
    'Defi_037_SunkenSecret': (26, 'Sunken Secret'),
    'Defi_032_WaterTrees': (27, 'Trees of Water'),
    'Defi_042_DeadlyCross': (28, 'Deadly Crossing'),
    'Defi_040_TheDescent': (30, 'The Descent'),
    # matched by elimination, then confirmed in play
    'Defi_046_Fire_3': (14, 'Hell'),
    'Defi_047_FireSheeper': (29, 'Wagons of Fire'),
    'Defi_005_X_Water': (15, 'Fluids'),
    'Defi_045_Aquasplash_2': (2, 'Running in the Rain'),
    'Defi_009_Collapse_NoPath': (22, 'Quicksand'),
}
CHALLENGE_GUESSED = set()


def title(name):
    """'Wildfires' for ALPHA_Map6_Fire, 'Fire!' for Defi_019_Fire; None if unknown. Demo files get
    '(demo)' appended."""
    m = re.match(r'ALPHA_Map(\d+)_', name or '')
    if m:
        t = STORY.get(int(m.group(1)))
        return t + (' (demo)' if t and 'Demo' in name else '') if t else None
    c = CHALLENGE.get(name)
    return c[1] + ('?' if name in CHALLENGE_GUESSED else '') if c else None


def label(name):
    """Display label for lists: 'Story 6: Wildfires', 'Challenge 1: Fire!' (file name if unknown)."""
    m = re.match(r'ALPHA_Map(\d+)_', name or '')
    t = title(name)
    if m and t:
        return 'Story %s: %s' % (m.group(1), t)
    if name in CHALLENGE:
        return 'Challenge %d: %s' % (CHALLENGE[name][0], t)
    return name
