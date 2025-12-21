use std::{convert::TryFrom, fmt};

/// Represents various keyboard layouts identified by a u32 code.
/// The codes correspond to standard keyboard layout identifiers used by Windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)] // Optional: Helps associate the discriminant, but TryFrom is more explicit
pub enum KeyboardLayout {
    /// ADLaM: 00140C00
    ADLaM = 0x00140C00,
    /// Albanian: 0000041C
    Albanian = 0x0000041C,
    /// Arabic (101): 00000401
    Arabic101 = 0x00000401,
    /// Arabic (102): 00010401
    Arabic102 = 0x00010401,
    /// Arabic (102) AZERTY: 00020401
    Arabic102Azerty = 0x00020401,
    /// Armenian Eastern (Legacy): 0000042B
    ArmenianEasternLegacy = 0x0000042B,
    /// Armenian Phonetic: 0002042B
    ArmenianPhonetic = 0x0002042B,
    /// Armenian Typewriter: 0003042B
    ArmenianTypewriter = 0x0003042B,
    /// Armenian Western (Legacy): 0001042B
    ArmenianWesternLegacy = 0x0001042B,
    /// Assamese - INSCRIPT: 0000044D
    AssameseInscript = 0x0000044D,
    /// Azerbaijani (Standard): 0001042C
    AzerbaijaniStandard = 0x0001042C,
    /// Azerbaijani Cyrillic: 0000082C
    AzerbaijaniCyrillic = 0x0000082C,
    /// Azerbaijani Latin: 0000042C
    AzerbaijaniLatin = 0x0000042C,
    /// Bangla: 00000445
    Bangla = 0x00000445,
    /// Bangla - INSCRIPT: 00020445
    BanglaInscript = 0x00020445,
    /// Bangla - INSCRIPT (Legacy): 00010445
    BanglaInscriptLegacy = 0x00010445,
    /// Bashkir: 0000046D
    Bashkir = 0x0000046D,
    /// Belarusian: 00000423
    Belarusian = 0x00000423,
    /// Belgian (Comma): 0001080C
    BelgianComma = 0x0001080C,
    /// Belgian (Period): 00000813
    BelgianPeriod = 0x00000813,
    /// Belgian French: 0000080C
    BelgianFrench = 0x0000080C,
    /// Bosnian (Cyrillic): 0000201A
    BosnianCyrillic = 0x0000201A,
    /// Buginese: 000B0C00
    Buginese = 0x000B0C00,
    /// Bulgarian: 00030402
    Bulgarian = 0x00030402,
    /// Bulgarian (Latin): 00010402
    BulgarianLatin = 0x00010402,
    /// Bulgarian (Phonetic Traditional): 00040402
    BulgarianPhoneticTraditional = 0x00040402,
    /// Bulgarian (Phonetic): 00020402
    BulgarianPhonetic = 0x00020402,
    /// Bulgarian (Typewriter): 00000402
    BulgarianTypewriter = 0x00000402,
    /// Canadian French: 00001009
    CanadianFrench = 0x00001009,
    /// Canadian French (Legacy): 00000C0C
    CanadianFrenchLegacy = 0x00000C0C,
    /// Canadian Multilingual Standard: 00011009
    CanadianMultilingualStandard = 0x00011009,
    /// Central Atlas Tamazight: 0000085F
    CentralAtlasTamazight = 0x0000085F,
    /// Central Kurdish: 00000492
    CentralKurdish = 0x00000492,
    /// Cherokee Nation: 0000045C
    CherokeeNation = 0x0000045C,
    /// Cherokee Phonetic: 0001045C
    CherokeePhonetic = 0x0001045C,
    /// Chinese (Simplified) - US: 00000804
    ChineseSimplifiedUS = 0x00000804,
    /// Chinese (Simplified, Singapore) - US: 00001004
    ChineseSimplifiedSingaporeUS = 0x00001004,
    /// Chinese (Traditional) - US: 00000404
    ChineseTraditionalUS = 0x00000404,
    /// Chinese (Traditional, Hong Kong S.A.R.) - US: 00000C04
    ChineseTraditionalHongKongSARUS = 0x00000C04,
    /// Chinese (Traditional, Macao S.A.R.) - US: 00001404
    ChineseTraditionalMacaoSARUS = 0x00001404,
    /// Czech: 00000405
    Czech = 0x00000405,
    /// Czech (QWERTY): 00010405
    CzechQwerty = 0x00010405,
    /// Czech Programmers: 00020405
    CzechProgrammers = 0x00020405,
    /// Danish: 00000406
    Danish = 0x00000406,
    /// Devanagari - INSCRIPT: 00000439
    DevanagariInscript = 0x00000439,
    /// Divehi Phonetic: 00000465
    DivehiPhonetic = 0x00000465,
    /// Divehi Typewriter: 00010465
    DivehiTypewriter = 0x00010465,
    /// Dutch: 00000413
    Dutch = 0x00000413,
    /// Dzongkha: 00000C51
    Dzongkha = 0x00000C51,
    /// English (India): 00004009
    EnglishIndia = 0x00004009,
    /// Estonian: 00000425
    Estonian = 0x00000425,
    /// Faeroese: 00000438
    Faeroese = 0x00000438,
    /// Finnish: 0000040B
    Finnish = 0x0000040B,
    /// Finnish with Sami: 0001083B
    FinnishWithSami = 0x0001083B,
    /// French: 0000040C
    French = 0x0000040C,
    /// Futhark: 00120C00
    Futhark = 0x00120C00,
    /// Georgian (Ergonomic): 00020437
    GeorgianErgonomic = 0x00020437,
    /// Georgian (Legacy): 00000437
    GeorgianLegacy = 0x00000437,
    /// Georgian (MES): 00030437
    GeorgianMes = 0x00030437,
    /// Georgian (Old Alphabets): 00040437
    GeorgianOldAlphabets = 0x00040437,
    /// Georgian (QWERTY): 00010437
    GeorgianQwerty = 0x00010437,
    /// German: 00000407
    German = 0x00000407,
    /// German (IBM): 00010407
    GermanIbm = 0x00010407,
    /// Gothic: 000C0C00
    Gothic = 0x000C0C00,
    /// Greek: 00000408
    Greek = 0x00000408,
    /// Greek (220): 00010408
    Greek220 = 0x00010408,
    /// Greek (220) Latin: 00030408
    Greek220Latin = 0x00030408,
    /// Greek (319): 00020408
    Greek319 = 0x00020408,
    /// Greek (319) Latin: 00040408
    Greek319Latin = 0x00040408,
    /// Greek Latin: 00050408
    GreekLatin = 0x00050408,
    /// Greek Polytonic: 00060408
    GreekPolytonic = 0x00060408,
    /// Greenlandic: 0000046F
    Greenlandic = 0x0000046F,
    /// Guarani: 00000474
    Guarani = 0x00000474,
    /// Gujarati: 00000447
    Gujarati = 0x00000447,
    /// Hausa: 00000468
    Hausa = 0x00000468,
    /// Hawaiian: 00000475
    Hawaiian = 0x00000475,
    /// Hebrew: 0000040D
    Hebrew = 0x0000040D,
    /// Hebrew (Standard): 0002040D
    HebrewStandard = 0x0002040D,
    /// Hindi Traditional: 00010439
    HindiTraditional = 0x00010439,
    /// Hungarian: 0000040E
    Hungarian = 0x0000040E,
    /// Hungarian 101-key: 0001040E
    Hungarian101Key = 0x0001040E,
    /// Icelandic: 0000040F
    Icelandic = 0x0000040F,
    /// Igbo: 00000470
    Igbo = 0x00000470,
    /// Inuktitut - Latin: 0000085D
    InuktitutLatin = 0x0000085D,
    /// Inuktitut - Naqittaut: 0001045D
    InuktitutNaqittaut = 0x0001045D,
    /// Irish: 00001809
    Irish = 0x00001809,
    /// Italian: 00000410
    Italian = 0x00000410,
    /// Italian (142): 00010410
    Italian142 = 0x00010410,
    /// Japanese: 00000411
    Japanese = 0x00000411,
    /// Javanese: 00110C00
    Javanese = 0x00110C00,
    /// Kannada: 0000044B
    Kannada = 0x0000044B,
    /// Kazakh: 0000043F
    Kazakh = 0x0000043F,
    /// Khmer: 00000453
    Khmer = 0x00000453,
    /// Khmer (NIDA): 00010453
    KhmerNida = 0x00010453,
    /// Korean: 00000412
    Korean = 0x00000412,
    /// Kyrgyz Cyrillic: 00000440
    KyrgyzCyrillic = 0x00000440,
    /// Lao: 00000454
    Lao = 0x00000454,
    /// Latin American: 0000080A
    LatinAmerican = 0x0000080A,
    /// Latvian: 00000426
    Latvian = 0x00000426,
    /// Latvian (QWERTY): 00010426
    LatvianQwerty = 0x00010426,
    /// Latvian (Standard): 00020426
    LatvianStandard = 0x00020426,
    /// Lisu (Basic): 00070C00
    LisuBasic = 0x00070C00,
    /// Lisu (Standard): 00080C00
    LisuStandard = 0x00080C00,
    /// Lithuanian: 00010427
    Lithuanian = 0x00010427,
    /// Lithuanian IBM: 00000427
    LithuanianIbm = 0x00000427,
    /// Lithuanian Standard: 00020427
    LithuanianStandard = 0x00020427,
    /// Luxembourgish: 0000046E
    Luxembourgish = 0x0000046E,
    /// Macedonian: 0000042F
    Macedonian = 0x0000042F,
    /// Macedonian - Standard: 0001042F
    MacedonianStandard = 0x0001042F,
    /// Malayalam: 0000044C
    Malayalam = 0x0000044C,
    /// Maltese 47-Key: 0000043A
    Maltese47Key = 0x0000043A,
    /// Maltese 48-Key: 0001043A
    Maltese48Key = 0x0001043A,
    /// Maori: 00000481
    Maori = 0x00000481,
    /// Marathi: 0000044E
    Marathi = 0x0000044E,
    /// Mongolian (Mongolian Script): 00000850
    MongolianMongolianScript = 0x00000850,
    /// Mongolian Cyrillic: 00000450
    MongolianCyrillic = 0x00000450,
    /// Myanmar (Phonetic order): 00010C00
    MyanmarPhoneticOrder = 0x00010C00,
    /// Myanmar (Visual order): 00130C00
    MyanmarVisualOrder = 0x00130C00,
    /// NZ Aotearoa: 00001409
    NZAotearoa = 0x00001409,
    /// Nepali: 00000461
    Nepali = 0x00000461,
    /// New Tai Lue: 00020C00
    NewTaiLue = 0x00020C00,
    /// Norwegian: 00000414
    Norwegian = 0x00000414,
    /// Norwegian with Sami: 0000043B
    NorwegianWithSami = 0x0000043B,
    /// N'Ko: 00090C00
    Nko = 0x00090C00,
    /// Odia: 00000448
    Odia = 0x00000448,
    /// Ogham: 00040C00
    Ogham = 0x00040C00,
    /// Ol Chiki: 000D0C00
    OlChiki = 0x000D0C00,
    /// Old Italic: 000F0C00
    OldItalic = 0x000F0C00,
    /// Osage: 00150C00
    Osage = 0x00150C00,
    /// Osmanya: 000E0C00
    Osmanya = 0x000E0C00,
    /// Pashto (Afghanistan): 00000463
    PashtoAfghanistan = 0x00000463,
    /// Persian: 00000429
    Persian = 0x00000429,
    /// Persian (Standard): 00050429
    PersianStandard = 0x00050429,
    /// Phags-pa: 000A0C00
    PhagsPa = 0x000A0C00,
    /// Polish (214): 00010415
    Polish214 = 0x00010415,
    /// Polish (Programmers): 00000415
    PolishProgrammers = 0x00000415,
    /// Portuguese: 00000816
    Portuguese = 0x00000816,
    /// Portuguese (Brazil ABNT): 00000416
    PortugueseBrazilABNT = 0x00000416,
    /// Portuguese (Brazil ABNT2): 00010416
    PortugueseBrazilABNT2 = 0x00010416,
    /// Punjabi: 00000446
    Punjabi = 0x00000446,
    /// Romanian (Legacy): 00000418
    RomanianLegacy = 0x00000418,
    /// Romanian (Programmers): 00020418
    RomanianProgrammers = 0x00020418,
    /// Romanian (Standard): 00010418
    RomanianStandard = 0x00010418,
    /// Russian: 00000419
    Russian = 0x00000419,
    /// Russian (Typewriter): 00010419
    RussianTypewriter = 0x00010419,
    /// Russian - Mnemonic: 00020419
    RussianMnemonic = 0x00020419,
    /// Sakha: 00000485
    Sakha = 0x00000485,
    /// Sami Extended Finland-Sweden: 0002083B
    SamiExtendedFinlandSweden = 0x0002083B,
    /// Sami Extended Norway: 0001043B
    SamiExtendedNorway = 0x0001043B,
    /// Scottish Gaelic: 00011809
    ScottishGaelic = 0x00011809,
    /// Serbian (Cyrillic): 00000C1A
    SerbianCyrillic = 0x00000C1A,
    /// Serbian (Latin): 0000081A
    SerbianLatin = 0x0000081A,
    /// Sesotho sa Leboa: 0000046C
    SesothoSaLeboa = 0x0000046C,
    /// Setswana: 00000432
    Setswana = 0x00000432,
    /// Sinhala: 0000045B
    Sinhala = 0x0000045B,
    /// Sinhala - Wij 9: 0001045B
    SinhalaWij9 = 0x0001045B,
    /// Slovak: 0000041B
    Slovak = 0x0000041B,
    /// Slovak (QWERTY): 0001041B
    SlovakQwerty = 0x0001041B,
    /// Slovenian: 00000424
    Slovenian = 0x00000424,
    /// Sora: 00100C00
    Sora = 0x00100C00,
    /// Sorbian Extended: 0001042E
    SorbianExtended = 0x0001042E,
    /// Sorbian Standard: 0002042E
    SorbianStandard = 0x0002042E,
    /// Sorbian Standard (Legacy): 0000042E
    SorbianStandardLegacy = 0x0000042E,
    /// Spanish: 0000040A
    Spanish = 0x0000040A,
    /// Spanish Variation: 0001040A
    SpanishVariation = 0x0001040A,
    /// Standard: 0000041A  // Note: "Standard" might be ambiguous, likely refers to a specific context (e.g., Serbian/Croatian Standard)
    Standard = 0x0000041A,
    /// Swedish: 0000041D
    Swedish = 0x0000041D,
    /// Swedish with Sami: 0000083B
    SwedishWithSami = 0x0000083B,
    /// Swiss French: 0000100C
    SwissFrench = 0x0000100C,
    /// Swiss German: 00000807
    SwissGerman = 0x00000807,
    /// Syriac: 0000045A
    Syriac = 0x0000045A,
    /// Syriac Phonetic: 0001045A
    SyriacPhonetic = 0x0001045A,
    /// Tai Le: 00030C00
    TaiLe = 0x00030C00,
    /// Tajik: 00000428
    Tajik = 0x00000428,
    /// Tamil: 00000449
    Tamil = 0x00000449,
    /// Tamil 99: 00020449
    Tamil99 = 0x00020449,
    /// Tamil Anjal: 00030449
    TamilAnjal = 0x00030449,
    /// Tatar: 00010444
    Tatar = 0x00010444,
    /// Tatar (Legacy): 00000444
    TatarLegacy = 0x00000444,
    /// Telugu: 0000044A
    Telugu = 0x0000044A,
    /// Thai Kedmanee: 0000041E
    ThaiKedmanee = 0x0000041E,
    /// Thai Kedmanee (non-ShiftLock): 0002041E
    ThaiKedmaneeNonShiftLock = 0x0002041E,
    /// Thai Pattachote: 0001041E
    ThaiPattachote = 0x0001041E,
    /// Thai Pattachote (non-ShiftLock): 0003041E
    ThaiPattachoteNonShiftLock = 0x0003041E,
    /// Tibetan (PRC): 00000451
    TibetanPRC = 0x00000451,
    /// Tibetan (PRC) - Updated: 00010451
    TibetanPRCUpdated = 0x00010451,
    /// Tifinagh (Basic): 0000105F
    TifinaghBasic = 0x0000105F,
    /// Tifinagh (Extended): 0001105F
    TifinaghExtended = 0x0001105F,
    /// Traditional Mongolian (Standard): 00010850
    TraditionalMongolianStandard = 0x00010850,
    /// Turkish F: 0001041F
    TurkishF = 0x0001041F,
    /// Turkish Q: 0000041F
    TurkishQ = 0x0000041F,
    /// Turkmen: 00000442
    Turkmen = 0x00000442,
    /// US: 00000409
    US = 0x00000409,
    /// US English Table for IBM Arabic 238_L: 00050409
    USEnglishTableForIBMArabic238L = 0x00050409,
    /// Ukrainian: 00000422
    Ukrainian = 0x00000422,
    /// Ukrainian (Enhanced): 00020422
    UkrainianEnhanced = 0x00020422,
    /// United Kingdom: 00000809
    UnitedKingdom = 0x00000809,
    /// United Kingdom Extended: 00000452
    UnitedKingdomExtended = 0x00000452,
    /// United States-Dvorak: 00010409
    UnitedStatesDvorak = 0x00010409,
    /// United States-Dvorak for left hand: 00030409
    UnitedStatesDvorakLeftHand = 0x00030409,
    /// United States-Dvorak for right hand: 00040409
    UnitedStatesDvorakRightHand = 0x00040409,
    /// United States-International: 00020409
    UnitedStatesInternational = 0x00020409,
    /// Urdu: 00000420
    Urdu = 0x00000420,
    /// Uyghur: 00010480
    Uyghur = 0x00010480,
    /// Uyghur (Legacy): 00000480
    UyghurLegacy = 0x00000480,
    /// Uzbek Cyrillic: 00000843
    UzbekCyrillic = 0x00000843,
    /// Vietnamese: 0000042A
    Vietnamese = 0x0000042A,
    /// Wolof: 00000488
    Wolof = 0x00000488,
    /// Yoruba: 0000046A
    Yoruba = 0x0000046A,
}

/// Error type for conversion from u32 to KeyboardLayout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnknownKeyboardLayoutError(pub u32);

impl std::fmt::Display for UnknownKeyboardLayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unknown keyboard layout identifier: 0x{:08X}", self.0)
    }
}

impl std::error::Error for UnknownKeyboardLayoutError {}

impl TryFrom<u32> for KeyboardLayout {
    type Error = UnknownKeyboardLayoutError;

    /// Attempts to convert a u32 identifier into a KeyboardLayout.
    /// Returns an error if the identifier is not recognized.
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0x00140C00 => Ok(KeyboardLayout::ADLaM),
            0x0000041C => Ok(KeyboardLayout::Albanian),
            0x00000401 => Ok(KeyboardLayout::Arabic101),
            0x00010401 => Ok(KeyboardLayout::Arabic102),
            0x00020401 => Ok(KeyboardLayout::Arabic102Azerty),
            0x0000042B => Ok(KeyboardLayout::ArmenianEasternLegacy),
            0x0002042B => Ok(KeyboardLayout::ArmenianPhonetic),
            0x0003042B => Ok(KeyboardLayout::ArmenianTypewriter),
            0x0001042B => Ok(KeyboardLayout::ArmenianWesternLegacy),
            0x0000044D => Ok(KeyboardLayout::AssameseInscript),
            0x0001042C => Ok(KeyboardLayout::AzerbaijaniStandard),
            0x0000082C => Ok(KeyboardLayout::AzerbaijaniCyrillic),
            0x0000042C => Ok(KeyboardLayout::AzerbaijaniLatin),
            0x00000445 => Ok(KeyboardLayout::Bangla),
            0x00020445 => Ok(KeyboardLayout::BanglaInscript),
            0x00010445 => Ok(KeyboardLayout::BanglaInscriptLegacy),
            0x0000046D => Ok(KeyboardLayout::Bashkir),
            0x00000423 => Ok(KeyboardLayout::Belarusian),
            0x0001080C => Ok(KeyboardLayout::BelgianComma),
            0x00000813 => Ok(KeyboardLayout::BelgianPeriod),
            0x0000080C => Ok(KeyboardLayout::BelgianFrench),
            0x0000201A => Ok(KeyboardLayout::BosnianCyrillic),
            0x000B0C00 => Ok(KeyboardLayout::Buginese),
            0x00030402 => Ok(KeyboardLayout::Bulgarian),
            0x00010402 => Ok(KeyboardLayout::BulgarianLatin),
            0x00040402 => Ok(KeyboardLayout::BulgarianPhoneticTraditional),
            0x00020402 => Ok(KeyboardLayout::BulgarianPhonetic),
            0x00000402 => Ok(KeyboardLayout::BulgarianTypewriter),
            0x00001009 => Ok(KeyboardLayout::CanadianFrench),
            0x00000C0C => Ok(KeyboardLayout::CanadianFrenchLegacy),
            0x00011009 => Ok(KeyboardLayout::CanadianMultilingualStandard),
            0x0000085F => Ok(KeyboardLayout::CentralAtlasTamazight),
            0x00000492 => Ok(KeyboardLayout::CentralKurdish),
            0x0000045C => Ok(KeyboardLayout::CherokeeNation),
            0x0001045C => Ok(KeyboardLayout::CherokeePhonetic),
            0x00000804 => Ok(KeyboardLayout::ChineseSimplifiedUS),
            0x00001004 => Ok(KeyboardLayout::ChineseSimplifiedSingaporeUS),
            0x00000404 => Ok(KeyboardLayout::ChineseTraditionalUS),
            0x00000C04 => Ok(KeyboardLayout::ChineseTraditionalHongKongSARUS),
            0x00001404 => Ok(KeyboardLayout::ChineseTraditionalMacaoSARUS),
            0x00000405 => Ok(KeyboardLayout::Czech),
            0x00010405 => Ok(KeyboardLayout::CzechQwerty),
            0x00020405 => Ok(KeyboardLayout::CzechProgrammers),
            0x00000406 => Ok(KeyboardLayout::Danish),
            0x00000439 => Ok(KeyboardLayout::DevanagariInscript),
            0x00000465 => Ok(KeyboardLayout::DivehiPhonetic),
            0x00010465 => Ok(KeyboardLayout::DivehiTypewriter),
            0x00000413 => Ok(KeyboardLayout::Dutch),
            0x00000C51 => Ok(KeyboardLayout::Dzongkha),
            0x00004009 => Ok(KeyboardLayout::EnglishIndia),
            0x00000425 => Ok(KeyboardLayout::Estonian),
            0x00000438 => Ok(KeyboardLayout::Faeroese),
            0x0000040B => Ok(KeyboardLayout::Finnish),
            0x0001083B => Ok(KeyboardLayout::FinnishWithSami),
            0x0000040C => Ok(KeyboardLayout::French),
            0x00120C00 => Ok(KeyboardLayout::Futhark),
            0x00020437 => Ok(KeyboardLayout::GeorgianErgonomic),
            0x00000437 => Ok(KeyboardLayout::GeorgianLegacy),
            0x00030437 => Ok(KeyboardLayout::GeorgianMes),
            0x00040437 => Ok(KeyboardLayout::GeorgianOldAlphabets),
            0x00010437 => Ok(KeyboardLayout::GeorgianQwerty),
            0x00000407 => Ok(KeyboardLayout::German),
            0x00010407 => Ok(KeyboardLayout::GermanIbm),
            0x000C0C00 => Ok(KeyboardLayout::Gothic),
            0x00000408 => Ok(KeyboardLayout::Greek),
            0x00010408 => Ok(KeyboardLayout::Greek220),
            0x00030408 => Ok(KeyboardLayout::Greek220Latin),
            0x00020408 => Ok(KeyboardLayout::Greek319),
            0x00040408 => Ok(KeyboardLayout::Greek319Latin),
            0x00050408 => Ok(KeyboardLayout::GreekLatin),
            0x00060408 => Ok(KeyboardLayout::GreekPolytonic),
            0x0000046F => Ok(KeyboardLayout::Greenlandic),
            0x00000474 => Ok(KeyboardLayout::Guarani),
            0x00000447 => Ok(KeyboardLayout::Gujarati),
            0x00000468 => Ok(KeyboardLayout::Hausa),
            0x00000475 => Ok(KeyboardLayout::Hawaiian),
            0x0000040D => Ok(KeyboardLayout::Hebrew),
            0x0002040D => Ok(KeyboardLayout::HebrewStandard),
            0x00010439 => Ok(KeyboardLayout::HindiTraditional),
            0x0000040E => Ok(KeyboardLayout::Hungarian),
            0x0001040E => Ok(KeyboardLayout::Hungarian101Key),
            0x0000040F => Ok(KeyboardLayout::Icelandic),
            0x00000470 => Ok(KeyboardLayout::Igbo),
            0x0000085D => Ok(KeyboardLayout::InuktitutLatin),
            0x0001045D => Ok(KeyboardLayout::InuktitutNaqittaut),
            0x00001809 => Ok(KeyboardLayout::Irish),
            0x00000410 => Ok(KeyboardLayout::Italian),
            0x00010410 => Ok(KeyboardLayout::Italian142),
            0x00000411 => Ok(KeyboardLayout::Japanese),
            0x00110C00 => Ok(KeyboardLayout::Javanese),
            0x0000044B => Ok(KeyboardLayout::Kannada),
            0x0000043F => Ok(KeyboardLayout::Kazakh),
            0x00000453 => Ok(KeyboardLayout::Khmer),
            0x00010453 => Ok(KeyboardLayout::KhmerNida),
            0x00000412 => Ok(KeyboardLayout::Korean),
            0x00000440 => Ok(KeyboardLayout::KyrgyzCyrillic),
            0x00000454 => Ok(KeyboardLayout::Lao),
            0x0000080A => Ok(KeyboardLayout::LatinAmerican),
            0x00000426 => Ok(KeyboardLayout::Latvian),
            0x00010426 => Ok(KeyboardLayout::LatvianQwerty),
            0x00020426 => Ok(KeyboardLayout::LatvianStandard),
            0x00070C00 => Ok(KeyboardLayout::LisuBasic),
            0x00080C00 => Ok(KeyboardLayout::LisuStandard),
            0x00010427 => Ok(KeyboardLayout::Lithuanian),
            0x00000427 => Ok(KeyboardLayout::LithuanianIbm),
            0x00020427 => Ok(KeyboardLayout::LithuanianStandard),
            0x0000046E => Ok(KeyboardLayout::Luxembourgish),
            0x0000042F => Ok(KeyboardLayout::Macedonian),
            0x0001042F => Ok(KeyboardLayout::MacedonianStandard),
            0x0000044C => Ok(KeyboardLayout::Malayalam),
            0x0000043A => Ok(KeyboardLayout::Maltese47Key),
            0x0001043A => Ok(KeyboardLayout::Maltese48Key),
            0x00000481 => Ok(KeyboardLayout::Maori),
            0x0000044E => Ok(KeyboardLayout::Marathi),
            0x00000850 => Ok(KeyboardLayout::MongolianMongolianScript),
            0x00000450 => Ok(KeyboardLayout::MongolianCyrillic),
            0x00010C00 => Ok(KeyboardLayout::MyanmarPhoneticOrder),
            0x00130C00 => Ok(KeyboardLayout::MyanmarVisualOrder),
            0x00001409 => Ok(KeyboardLayout::NZAotearoa),
            0x00000461 => Ok(KeyboardLayout::Nepali),
            0x00020C00 => Ok(KeyboardLayout::NewTaiLue),
            0x00000414 => Ok(KeyboardLayout::Norwegian),
            0x0000043B => Ok(KeyboardLayout::NorwegianWithSami),
            0x00090C00 => Ok(KeyboardLayout::Nko),
            0x00000448 => Ok(KeyboardLayout::Odia),
            0x00040C00 => Ok(KeyboardLayout::Ogham),
            0x000D0C00 => Ok(KeyboardLayout::OlChiki),
            0x000F0C00 => Ok(KeyboardLayout::OldItalic),
            0x00150C00 => Ok(KeyboardLayout::Osage),
            0x000E0C00 => Ok(KeyboardLayout::Osmanya),
            0x00000463 => Ok(KeyboardLayout::PashtoAfghanistan),
            0x00000429 => Ok(KeyboardLayout::Persian),
            0x00050429 => Ok(KeyboardLayout::PersianStandard),
            0x000A0C00 => Ok(KeyboardLayout::PhagsPa),
            0x00010415 => Ok(KeyboardLayout::Polish214),
            0x00000415 => Ok(KeyboardLayout::PolishProgrammers),
            0x00000816 => Ok(KeyboardLayout::Portuguese),
            0x00000416 => Ok(KeyboardLayout::PortugueseBrazilABNT),
            0x00010416 => Ok(KeyboardLayout::PortugueseBrazilABNT2),
            0x00000446 => Ok(KeyboardLayout::Punjabi),
            0x00000418 => Ok(KeyboardLayout::RomanianLegacy),
            0x00020418 => Ok(KeyboardLayout::RomanianProgrammers),
            0x00010418 => Ok(KeyboardLayout::RomanianStandard),
            0x00000419 => Ok(KeyboardLayout::Russian),
            0x00010419 => Ok(KeyboardLayout::RussianTypewriter),
            0x00020419 => Ok(KeyboardLayout::RussianMnemonic),
            0x00000485 => Ok(KeyboardLayout::Sakha),
            0x0002083B => Ok(KeyboardLayout::SamiExtendedFinlandSweden),
            0x0001043B => Ok(KeyboardLayout::SamiExtendedNorway),
            0x00011809 => Ok(KeyboardLayout::ScottishGaelic),
            0x00000C1A => Ok(KeyboardLayout::SerbianCyrillic),
            0x0000081A => Ok(KeyboardLayout::SerbianLatin),
            0x0000046C => Ok(KeyboardLayout::SesothoSaLeboa),
            0x00000432 => Ok(KeyboardLayout::Setswana),
            0x0000045B => Ok(KeyboardLayout::Sinhala),
            0x0001045B => Ok(KeyboardLayout::SinhalaWij9),
            0x0000041B => Ok(KeyboardLayout::Slovak),
            0x0001041B => Ok(KeyboardLayout::SlovakQwerty),
            0x00000424 => Ok(KeyboardLayout::Slovenian),
            0x00100C00 => Ok(KeyboardLayout::Sora),
            0x0001042E => Ok(KeyboardLayout::SorbianExtended),
            0x0002042E => Ok(KeyboardLayout::SorbianStandard),
            0x0000042E => Ok(KeyboardLayout::SorbianStandardLegacy),
            0x0000040A => Ok(KeyboardLayout::Spanish),
            0x0001040A => Ok(KeyboardLayout::SpanishVariation),
            0x0000041A => Ok(KeyboardLayout::Standard),
            0x0000041D => Ok(KeyboardLayout::Swedish),
            0x0000083B => Ok(KeyboardLayout::SwedishWithSami),
            0x0000100C => Ok(KeyboardLayout::SwissFrench),
            0x00000807 => Ok(KeyboardLayout::SwissGerman),
            0x0000045A => Ok(KeyboardLayout::Syriac),
            0x0001045A => Ok(KeyboardLayout::SyriacPhonetic),
            0x00030C00 => Ok(KeyboardLayout::TaiLe),
            0x00000428 => Ok(KeyboardLayout::Tajik),
            0x00000449 => Ok(KeyboardLayout::Tamil),
            0x00020449 => Ok(KeyboardLayout::Tamil99),
            0x00030449 => Ok(KeyboardLayout::TamilAnjal),
            0x00010444 => Ok(KeyboardLayout::Tatar),
            0x00000444 => Ok(KeyboardLayout::TatarLegacy),
            0x0000044A => Ok(KeyboardLayout::Telugu),
            0x0000041E => Ok(KeyboardLayout::ThaiKedmanee),
            0x0002041E => Ok(KeyboardLayout::ThaiKedmaneeNonShiftLock),
            0x0001041E => Ok(KeyboardLayout::ThaiPattachote),
            0x0003041E => Ok(KeyboardLayout::ThaiPattachoteNonShiftLock),
            0x00000451 => Ok(KeyboardLayout::TibetanPRC),
            0x00010451 => Ok(KeyboardLayout::TibetanPRCUpdated),
            0x0000105F => Ok(KeyboardLayout::TifinaghBasic),
            0x0001105F => Ok(KeyboardLayout::TifinaghExtended),
            0x00010850 => Ok(KeyboardLayout::TraditionalMongolianStandard),
            0x0001041F => Ok(KeyboardLayout::TurkishF),
            0x0000041F => Ok(KeyboardLayout::TurkishQ),
            0x00000442 => Ok(KeyboardLayout::Turkmen),
            0x00000409 => Ok(KeyboardLayout::US),
            0x00050409 => Ok(KeyboardLayout::USEnglishTableForIBMArabic238L),
            0x00000422 => Ok(KeyboardLayout::Ukrainian),
            0x00020422 => Ok(KeyboardLayout::UkrainianEnhanced),
            0x00000809 => Ok(KeyboardLayout::UnitedKingdom),
            0x00000452 => Ok(KeyboardLayout::UnitedKingdomExtended),
            0x00010409 => Ok(KeyboardLayout::UnitedStatesDvorak),
            0x00030409 => Ok(KeyboardLayout::UnitedStatesDvorakLeftHand),
            0x00040409 => Ok(KeyboardLayout::UnitedStatesDvorakRightHand),
            0x00020409 => Ok(KeyboardLayout::UnitedStatesInternational),
            0x00000420 => Ok(KeyboardLayout::Urdu),
            0x00010480 => Ok(KeyboardLayout::Uyghur),
            0x00000480 => Ok(KeyboardLayout::UyghurLegacy),
            0x00000843 => Ok(KeyboardLayout::UzbekCyrillic),
            0x0000042A => Ok(KeyboardLayout::Vietnamese),
            0x00000488 => Ok(KeyboardLayout::Wolof),
            0x0000046A => Ok(KeyboardLayout::Yoruba),
            // Add other layouts here...
            unknown => Err(UnknownKeyboardLayoutError(unknown)),
        }
    }
}

// Implement the Display trait for KeyboardLayout
impl fmt::Display for KeyboardLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use the original human-readable names for display
        match self {
            KeyboardLayout::ADLaM => write!(f, "ADLaM"),
            KeyboardLayout::Albanian => write!(f, "Albanian"),
            KeyboardLayout::Arabic101 => write!(f, "Arabic (101)"),
            KeyboardLayout::Arabic102 => write!(f, "Arabic (102)"),
            KeyboardLayout::Arabic102Azerty => write!(f, "Arabic (102) AZERTY"),
            KeyboardLayout::ArmenianEasternLegacy => write!(f, "Armenian Eastern (Legacy)"),
            KeyboardLayout::ArmenianPhonetic => write!(f, "Armenian Phonetic"),
            KeyboardLayout::ArmenianTypewriter => write!(f, "Armenian Typewriter"),
            KeyboardLayout::ArmenianWesternLegacy => write!(f, "Armenian Western (Legacy)"),
            KeyboardLayout::AssameseInscript => write!(f, "Assamese - INSCRIPT"),
            KeyboardLayout::AzerbaijaniStandard => write!(f, "Azerbaijani (Standard)"),
            KeyboardLayout::AzerbaijaniCyrillic => write!(f, "Azerbaijani Cyrillic"),
            KeyboardLayout::AzerbaijaniLatin => write!(f, "Azerbaijani Latin"),
            KeyboardLayout::Bangla => write!(f, "Bangla"),
            KeyboardLayout::BanglaInscript => write!(f, "Bangla - INSCRIPT"),
            KeyboardLayout::BanglaInscriptLegacy => write!(f, "Bangla - INSCRIPT (Legacy)"),
            KeyboardLayout::Bashkir => write!(f, "Bashkir"),
            KeyboardLayout::Belarusian => write!(f, "Belarusian"),
            KeyboardLayout::BelgianComma => write!(f, "Belgian (Comma)"),
            KeyboardLayout::BelgianPeriod => write!(f, "Belgian (Period)"),
            KeyboardLayout::BelgianFrench => write!(f, "Belgian French"),
            KeyboardLayout::BosnianCyrillic => write!(f, "Bosnian (Cyrillic)"),
            KeyboardLayout::Buginese => write!(f, "Buginese"),
            KeyboardLayout::Bulgarian => write!(f, "Bulgarian"),
            KeyboardLayout::BulgarianLatin => write!(f, "Bulgarian (Latin)"),
            KeyboardLayout::BulgarianPhoneticTraditional => {
                write!(f, "Bulgarian (Phonetic Traditional)")
            }
            KeyboardLayout::BulgarianPhonetic => write!(f, "Bulgarian (Phonetic)"),
            KeyboardLayout::BulgarianTypewriter => write!(f, "Bulgarian (Typewriter)"),
            KeyboardLayout::CanadianFrench => write!(f, "Canadian French"),
            KeyboardLayout::CanadianFrenchLegacy => write!(f, "Canadian French (Legacy)"),
            KeyboardLayout::CanadianMultilingualStandard => {
                write!(f, "Canadian Multilingual Standard")
            }
            KeyboardLayout::CentralAtlasTamazight => write!(f, "Central Atlas Tamazight"),
            KeyboardLayout::CentralKurdish => write!(f, "Central Kurdish"),
            KeyboardLayout::CherokeeNation => write!(f, "Cherokee Nation"),
            KeyboardLayout::CherokeePhonetic => write!(f, "Cherokee Phonetic"),
            KeyboardLayout::ChineseSimplifiedUS => write!(f, "Chinese (Simplified) - US"),
            KeyboardLayout::ChineseSimplifiedSingaporeUS => {
                write!(f, "Chinese (Simplified, Singapore) - US")
            }
            KeyboardLayout::ChineseTraditionalUS => write!(f, "Chinese (Traditional) - US"),
            KeyboardLayout::ChineseTraditionalHongKongSARUS => {
                write!(f, "Chinese (Traditional, Hong Kong S.A.R.) - US")
            }
            KeyboardLayout::ChineseTraditionalMacaoSARUS => {
                write!(f, "Chinese (Traditional, Macao S.A.R.) - US")
            }
            KeyboardLayout::Czech => write!(f, "Czech"),
            KeyboardLayout::CzechQwerty => write!(f, "Czech (QWERTY)"),
            KeyboardLayout::CzechProgrammers => write!(f, "Czech Programmers"),
            KeyboardLayout::Danish => write!(f, "Danish"),
            KeyboardLayout::DevanagariInscript => write!(f, "Devanagari - INSCRIPT"),
            KeyboardLayout::DivehiPhonetic => write!(f, "Divehi Phonetic"),
            KeyboardLayout::DivehiTypewriter => write!(f, "Divehi Typewriter"),
            KeyboardLayout::Dutch => write!(f, "Dutch"),
            KeyboardLayout::Dzongkha => write!(f, "Dzongkha"),
            KeyboardLayout::EnglishIndia => write!(f, "English (India)"),
            KeyboardLayout::Estonian => write!(f, "Estonian"),
            KeyboardLayout::Faeroese => write!(f, "Faeroese"),
            KeyboardLayout::Finnish => write!(f, "Finnish"),
            KeyboardLayout::FinnishWithSami => write!(f, "Finnish with Sami"),
            KeyboardLayout::French => write!(f, "French"),
            KeyboardLayout::Futhark => write!(f, "Futhark"),
            KeyboardLayout::GeorgianErgonomic => write!(f, "Georgian (Ergonomic)"),
            KeyboardLayout::GeorgianLegacy => write!(f, "Georgian (Legacy)"),
            KeyboardLayout::GeorgianMes => write!(f, "Georgian (MES)"),
            KeyboardLayout::GeorgianOldAlphabets => write!(f, "Georgian (Old Alphabets)"),
            KeyboardLayout::GeorgianQwerty => write!(f, "Georgian (QWERTY)"),
            KeyboardLayout::German => write!(f, "German"),
            KeyboardLayout::GermanIbm => write!(f, "German (IBM)"),
            KeyboardLayout::Gothic => write!(f, "Gothic"),
            KeyboardLayout::Greek => write!(f, "Greek"),
            KeyboardLayout::Greek220 => write!(f, "Greek (220)"),
            KeyboardLayout::Greek220Latin => write!(f, "Greek (220) Latin"),
            KeyboardLayout::Greek319 => write!(f, "Greek (319)"),
            KeyboardLayout::Greek319Latin => write!(f, "Greek (319) Latin"),
            KeyboardLayout::GreekLatin => write!(f, "Greek Latin"),
            KeyboardLayout::GreekPolytonic => write!(f, "Greek Polytonic"),
            KeyboardLayout::Greenlandic => write!(f, "Greenlandic"),
            KeyboardLayout::Guarani => write!(f, "Guarani"),
            KeyboardLayout::Gujarati => write!(f, "Gujarati"),
            KeyboardLayout::Hausa => write!(f, "Hausa"),
            KeyboardLayout::Hawaiian => write!(f, "Hawaiian"),
            KeyboardLayout::Hebrew => write!(f, "Hebrew"),
            KeyboardLayout::HebrewStandard => write!(f, "Hebrew (Standard)"),
            KeyboardLayout::HindiTraditional => write!(f, "Hindi Traditional"),
            KeyboardLayout::Hungarian => write!(f, "Hungarian"),
            KeyboardLayout::Hungarian101Key => write!(f, "Hungarian 101-key"),
            KeyboardLayout::Icelandic => write!(f, "Icelandic"),
            KeyboardLayout::Igbo => write!(f, "Igbo"),
            KeyboardLayout::InuktitutLatin => write!(f, "Inuktitut - Latin"),
            KeyboardLayout::InuktitutNaqittaut => write!(f, "Inuktitut - Naqittaut"),
            KeyboardLayout::Irish => write!(f, "Irish"),
            KeyboardLayout::Italian => write!(f, "Italian"),
            KeyboardLayout::Italian142 => write!(f, "Italian (142)"),
            KeyboardLayout::Japanese => write!(f, "Japanese"),
            KeyboardLayout::Javanese => write!(f, "Javanese"),
            KeyboardLayout::Kannada => write!(f, "Kannada"),
            KeyboardLayout::Kazakh => write!(f, "Kazakh"),
            KeyboardLayout::Khmer => write!(f, "Khmer"),
            KeyboardLayout::KhmerNida => write!(f, "Khmer (NIDA)"),
            KeyboardLayout::Korean => write!(f, "Korean"),
            KeyboardLayout::KyrgyzCyrillic => write!(f, "Kyrgyz Cyrillic"),
            KeyboardLayout::Lao => write!(f, "Lao"),
            KeyboardLayout::LatinAmerican => write!(f, "Latin American"),
            KeyboardLayout::Latvian => write!(f, "Latvian"),
            KeyboardLayout::LatvianQwerty => write!(f, "Latvian (QWERTY)"),
            KeyboardLayout::LatvianStandard => write!(f, "Latvian (Standard)"),
            KeyboardLayout::LisuBasic => write!(f, "Lisu (Basic)"),
            KeyboardLayout::LisuStandard => write!(f, "Lisu (Standard)"),
            KeyboardLayout::Lithuanian => write!(f, "Lithuanian"),
            KeyboardLayout::LithuanianIbm => write!(f, "Lithuanian IBM"),
            KeyboardLayout::LithuanianStandard => write!(f, "Lithuanian Standard"),
            KeyboardLayout::Luxembourgish => write!(f, "Luxembourgish"),
            KeyboardLayout::Macedonian => write!(f, "Macedonian"),
            KeyboardLayout::MacedonianStandard => write!(f, "Macedonian - Standard"),
            KeyboardLayout::Malayalam => write!(f, "Malayalam"),
            KeyboardLayout::Maltese47Key => write!(f, "Maltese 47-Key"),
            KeyboardLayout::Maltese48Key => write!(f, "Maltese 48-Key"),
            KeyboardLayout::Maori => write!(f, "Maori"),
            KeyboardLayout::Marathi => write!(f, "Marathi"),
            KeyboardLayout::MongolianMongolianScript => write!(f, "Mongolian (Mongolian Script)"),
            KeyboardLayout::MongolianCyrillic => write!(f, "Mongolian Cyrillic"),
            KeyboardLayout::MyanmarPhoneticOrder => write!(f, "Myanmar (Phonetic order)"),
            KeyboardLayout::MyanmarVisualOrder => write!(f, "Myanmar (Visual order)"),
            KeyboardLayout::NZAotearoa => write!(f, "NZ Aotearoa"),
            KeyboardLayout::Nepali => write!(f, "Nepali"),
            KeyboardLayout::NewTaiLue => write!(f, "New Tai Lue"),
            KeyboardLayout::Norwegian => write!(f, "Norwegian"),
            KeyboardLayout::NorwegianWithSami => write!(f, "Norwegian with Sami"),
            KeyboardLayout::Nko => write!(f, "N'Ko"),
            KeyboardLayout::Odia => write!(f, "Odia"),
            KeyboardLayout::Ogham => write!(f, "Ogham"),
            KeyboardLayout::OlChiki => write!(f, "Ol Chiki"),
            KeyboardLayout::OldItalic => write!(f, "Old Italic"),
            KeyboardLayout::Osage => write!(f, "Osage"),
            KeyboardLayout::Osmanya => write!(f, "Osmanya"),
            KeyboardLayout::PashtoAfghanistan => write!(f, "Pashto (Afghanistan)"),
            KeyboardLayout::Persian => write!(f, "Persian"),
            KeyboardLayout::PersianStandard => write!(f, "Persian (Standard)"),
            KeyboardLayout::PhagsPa => write!(f, "Phags-pa"),
            KeyboardLayout::Polish214 => write!(f, "Polish (214)"),
            KeyboardLayout::PolishProgrammers => write!(f, "Polish (Programmers)"),
            KeyboardLayout::Portuguese => write!(f, "Portuguese"),
            KeyboardLayout::PortugueseBrazilABNT => write!(f, "Portuguese (Brazil ABNT)"),
            KeyboardLayout::PortugueseBrazilABNT2 => write!(f, "Portuguese (Brazil ABNT2)"),
            KeyboardLayout::Punjabi => write!(f, "Punjabi"),
            KeyboardLayout::RomanianLegacy => write!(f, "Romanian (Legacy)"),
            KeyboardLayout::RomanianProgrammers => write!(f, "Romanian (Programmers)"),
            KeyboardLayout::RomanianStandard => write!(f, "Romanian (Standard)"),
            KeyboardLayout::Russian => write!(f, "Russian"),
            KeyboardLayout::RussianTypewriter => write!(f, "Russian (Typewriter)"),
            KeyboardLayout::RussianMnemonic => write!(f, "Russian - Mnemonic"),
            KeyboardLayout::Sakha => write!(f, "Sakha"),
            KeyboardLayout::SamiExtendedFinlandSweden => write!(f, "Sami Extended Finland-Sweden"),
            KeyboardLayout::SamiExtendedNorway => write!(f, "Sami Extended Norway"),
            KeyboardLayout::ScottishGaelic => write!(f, "Scottish Gaelic"),
            KeyboardLayout::SerbianCyrillic => write!(f, "Serbian (Cyrillic)"),
            KeyboardLayout::SerbianLatin => write!(f, "Serbian (Latin)"),
            KeyboardLayout::SesothoSaLeboa => write!(f, "Sesotho sa Leboa"),
            KeyboardLayout::Setswana => write!(f, "Setswana"),
            KeyboardLayout::Sinhala => write!(f, "Sinhala"),
            KeyboardLayout::SinhalaWij9 => write!(f, "Sinhala - Wij 9"),
            KeyboardLayout::Slovak => write!(f, "Slovak"),
            KeyboardLayout::SlovakQwerty => write!(f, "Slovak (QWERTY)"),
            KeyboardLayout::Slovenian => write!(f, "Slovenian"),
            KeyboardLayout::Sora => write!(f, "Sora"),
            KeyboardLayout::SorbianExtended => write!(f, "Sorbian Extended"),
            KeyboardLayout::SorbianStandard => write!(f, "Sorbian Standard"),
            KeyboardLayout::SorbianStandardLegacy => write!(f, "Sorbian Standard (Legacy)"),
            KeyboardLayout::Spanish => write!(f, "Spanish"),
            KeyboardLayout::SpanishVariation => write!(f, "Spanish Variation"),
            KeyboardLayout::Standard => write!(f, "Standard"), // As noted before, this name might be ambiguous
            KeyboardLayout::Swedish => write!(f, "Swedish"),
            KeyboardLayout::SwedishWithSami => write!(f, "Swedish with Sami"),
            KeyboardLayout::SwissFrench => write!(f, "Swiss French"),
            KeyboardLayout::SwissGerman => write!(f, "Swiss German"),
            KeyboardLayout::Syriac => write!(f, "Syriac"),
            KeyboardLayout::SyriacPhonetic => write!(f, "Syriac Phonetic"),
            KeyboardLayout::TaiLe => write!(f, "Tai Le"),
            KeyboardLayout::Tajik => write!(f, "Tajik"),
            KeyboardLayout::Tamil => write!(f, "Tamil"),
            KeyboardLayout::Tamil99 => write!(f, "Tamil 99"),
            KeyboardLayout::TamilAnjal => write!(f, "Tamil Anjal"),
            KeyboardLayout::Tatar => write!(f, "Tatar"),
            KeyboardLayout::TatarLegacy => write!(f, "Tatar (Legacy)"),
            KeyboardLayout::Telugu => write!(f, "Telugu"),
            KeyboardLayout::ThaiKedmanee => write!(f, "Thai Kedmanee"),
            KeyboardLayout::ThaiKedmaneeNonShiftLock => write!(f, "Thai Kedmanee (non-ShiftLock)"),
            KeyboardLayout::ThaiPattachote => write!(f, "Thai Pattachote"),
            KeyboardLayout::ThaiPattachoteNonShiftLock => {
                write!(f, "Thai Pattachote (non-ShiftLock)")
            }
            KeyboardLayout::TibetanPRC => write!(f, "Tibetan (PRC)"),
            KeyboardLayout::TibetanPRCUpdated => write!(f, "Tibetan (PRC) - Updated"),
            KeyboardLayout::TifinaghBasic => write!(f, "Tifinagh (Basic)"),
            KeyboardLayout::TifinaghExtended => write!(f, "Tifinagh (Extended)"),
            KeyboardLayout::TraditionalMongolianStandard => {
                write!(f, "Traditional Mongolian (Standard)")
            }
            KeyboardLayout::TurkishF => write!(f, "Turkish F"),
            KeyboardLayout::TurkishQ => write!(f, "Turkish Q"),
            KeyboardLayout::Turkmen => write!(f, "Turkmen"),
            KeyboardLayout::US => write!(f, "US"),
            KeyboardLayout::USEnglishTableForIBMArabic238L => {
                write!(f, "US English Table for IBM Arabic 238_L")
            }
            KeyboardLayout::Ukrainian => write!(f, "Ukrainian"),
            KeyboardLayout::UkrainianEnhanced => write!(f, "Ukrainian (Enhanced)"),
            KeyboardLayout::UnitedKingdom => write!(f, "United Kingdom"),
            KeyboardLayout::UnitedKingdomExtended => write!(f, "United Kingdom Extended"),
            KeyboardLayout::UnitedStatesDvorak => write!(f, "United States-Dvorak"),
            KeyboardLayout::UnitedStatesDvorakLeftHand => {
                write!(f, "United States-Dvorak for left hand")
            }
            KeyboardLayout::UnitedStatesDvorakRightHand => {
                write!(f, "United States-Dvorak for right hand")
            }
            KeyboardLayout::UnitedStatesInternational => write!(f, "United States-International"),
            KeyboardLayout::Urdu => write!(f, "Urdu"),
            KeyboardLayout::Uyghur => write!(f, "Uyghur"),
            KeyboardLayout::UyghurLegacy => write!(f, "Uyghur (Legacy)"),
            KeyboardLayout::UzbekCyrillic => write!(f, "Uzbek Cyrillic"),
            KeyboardLayout::Vietnamese => write!(f, "Vietnamese"),
            KeyboardLayout::Wolof => write!(f, "Wolof"),
            KeyboardLayout::Yoruba => write!(f, "Yoruba"),
        }
    }
}
