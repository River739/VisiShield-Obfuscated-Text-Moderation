import Levenshtein
import re

# 1. Databases & Dictionaries
STOP_WORDS = set([
    "this", "that", "it", "the", "is", "a", "of", "to", "in", "but", "and",
    "or", "on", "for", "with", "as", "was", "be", "dont", "do", "not", "up", "you", "are", "dog"
])

BANNED_WORDS_BASE = ["bitch", "shit", "fuck", "asshole"]

ACRONYMS_AND_SLANG = {
    "wtf": "fuck",
    "mf": "bitch",
    "mfer": "fuck",
    "stfu": "fuck",
    "gtfo": "fuck"
}

DICTIONARY_SAFE = set(ROBUST_SAFE).union(STOP_WORDS)

# 2. Unified Stage 1 Filter
def stage_1_text_filter(word):
    # Normalize to lowercase immediately
    word_lower = word.lower()

    # Clean out trailing punctuation
    word_cleaned = re.sub(r'[^\w!@#$*_+-]', '', word_lower)

    # CRITICAL FIX: Check Acronyms & Slang FIRST before short-word length filters
    if word_cleaned in ACRONYMS_AND_SLANG:
        return "BANNED (Blocked by Acronym/Slang Dictionary Match)"

    # Check 2: Short word check (allow unless it's a known shorthand)
    if len(word_cleaned) <= 2 and word_cleaned not in ["sh", "fc"]:
        return "SAFE (Too short to escalate)"

    # Check 3: Safe / Stop Word Bypass
    if word_cleaned in DICTIONARY_SAFE:
        return "SAFE (Bypassed via Stage 1 Dictionary)"

    # Check 4: Direct Banned Match
    if word_cleaned in BANNED_WORDS_BASE:
        return "BANNED (Blocked by Stage 1 Direct Match)"

    # Check 5: Visual Stemming (Suffix Stripping like -ing, -ed)
    suffixes = ['ing', 'ed', 'er', 's']
    base_word = word_cleaned
    for suffix in suffixes:
        if word_cleaned.endswith(suffix) and len(word_cleaned) > len(suffix) + 2:
            base_word = word_cleaned[:-len(suffix)]
            break

    if base_word in BANNED_WORDS_BASE:
        return "BANNED (Blocked by Suffix Stemming Match)"

    # Check 6: Levenshtein Distance Check on both cleaned and base words
    is_suspicious = False
    for banned in BANNED_WORDS_BASE:
        distance_orig = Levenshtein.distance(word_cleaned, banned)
        distance_base = Levenshtein.distance(base_word, banned)
        if (0 < distance_orig <= 2) or (0 < distance_base <= 2):
            is_suspicious = True
            break

    # Check 7: Leet-speak symbols
    obfuscation_indicators = set(['*', '!', '@', '$', '1', '0', '_', '+', '#'])
    has_obfuscation = any(char in obfuscation_indicators for char in word_cleaned)

    if is_suspicious or has_obfuscation:
        return "SUSPICIOUS"
    else:
        return "SAFE (No suspicious patterns found)"

import re
import cv2
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms

def preprocess_despace(comment_text):
    cleaned = re.sub(r"[.\-_]+", "", comment_text)
    words = cleaned.split()

    # Only trigger despace if there are 3 or more single-letter/two-letter words
    # forming a single hidden word (e.g., "F u c k"), avoiding short sentences like "u a mf"
    single_letters = [w for w in words if len(w) == 1]
    if len(single_letters) >= 3 and len(words) <= 5:
        merged_word = "".join(words)
        if 3 <= len(merged_word) <= 10:
            return [merged_word]

    return re.findall(r"[a-zA-Z0-9!@#$*_+-]+", comment_text)

def moderate_comment(comment_text):
    print(f"Original Comment: \"{comment_text}\"")

    # Run the advanced de-spacing preprocessor
    words = preprocess_despace(comment_text)

    censored_words = []

    for original_word in words:
        stage_1_result = stage_1_text_filter(original_word)

        if stage_1_result == "SUSPICIOUS":
            normalized_word = original_word.lower()

            # Suffix stripping logic for image rendering
            suffixes = ['ing', 'ed', 'er', 's']
            for suffix in suffixes:
                if normalized_word.endswith(suffix) and len(normalized_word) > len(suffix) + 2:
                    normalized_word = normalized_word[:-len(suffix)]
                    break

            img = Image.new("L", (192, 64), color=255)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((20, 20), normalized_word, fill=0, font=font)
            img_np = np.array(img)
            blurred_img = cv2.GaussianBlur(img_np, (7, 7), 0)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            input_tensor = transform(Image.fromarray(blurred_img)).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()

            if prob > 0.5:
                print(f" -> [!] Censoring '{original_word}' (Stage 2 flagged as '{normalized_word}' with {prob*100:.1f}% confidence)")
                censored_words.append("[CENSORED]")
            else:
                censored_words.append(original_word)
        elif "BANNED" in stage_1_result:
            print(f" -> [!] Censoring '{original_word}' (Stage 1 direct match)")
            censored_words.append("[CENSORED]")
        else:
            censored_words.append(original_word)

    moderated_text = " ".join(censored_words)
    print(f"Moderated Comment: \"{moderated_text}\"\n")
    print("-" * 50)
def moderate_comment(comment_text):
    print(f"Original Comment: \"{comment_text}\"")

    # 1. Use our de-spacing preprocessor
    words = preprocess_despace(comment_text)

    is_toxic = False
    flagged_reasons = []

    for original_word in words:
        stage_1_result = stage_1_text_filter(original_word)

        if stage_1_result == "SUSPICIOUS":
            normalized_word = original_word.lower()

            # Suffix stripping logic
            suffixes = ['ing', 'ed', 'er', 's']
            for suffix in suffixes:
                if normalized_word.endswith(suffix) and len(normalized_word) > len(suffix) + 2:
                    normalized_word = normalized_word[:-len(suffix)]
                    break

            # Generate image for CV model
            img = Image.new("L", (192, 64), color=255)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((20, 20), normalized_word, fill=0, font=font)
            img_np = np.array(img)
            blurred_img = cv2.GaussianBlur(img_np, (7, 7), 0)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            input_tensor = transform(Image.fromarray(blurred_img)).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()

            if prob > 0.5:
                flagged_reasons.append(f"Visual signature '{normalized_word}' flagged by CV ({prob*100:.1f}% confidence)")
                is_toxic = True

        elif "BANNED" in stage_1_result:
            flagged_reasons.append(f"Direct text match for '{original_word}' in Stage 1")
            is_toxic = True

    # Final System Verdict
    print("-" * 50)
    if is_toxic:
        print("❌ PIPELINE VERDICT: REJECTED (Failed Community Guidelines)")
        for reason in flagged_reasons:
            print(f"  └─ Reason: {reason}")
    else:
        print("✅ PIPELINE VERDICT: APPROVED")
    print("-" * 50 + "\n")