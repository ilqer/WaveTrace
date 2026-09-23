# Hardware Selection — Radio Front End for the Next Phase

Status: research rev 2, 2026-09-22. No purchase made. Nothing here is implemented.
Scope: the radio hardware that replaces the ESP32 mesh for the two candidate directions
(people counting, concealed metal). Rev 1 (2026-09-21) ranked 58 parts against a single
link topology; rev 2 re-researched every live candidate against datasheets, adds the parts
rev 1 missed, and applies three decisions taken on 2026-09-22:

- **Budget: under USD 3,000** for six nodes including host boards and antennas.
- **Two topologies are ranked**, not one: (A) a cross-node link, as the ESP32 mesh works
  today, and (B) six independent monostatic radars. Which one to build is still open.
- **Band: 2-20 GHz is the goal.** Licence-free 60 GHz parts are ranked as a secondary
  list because they are cheap and wideband, with the caveat that 60-70 GHz concealed-object
  work already exists (mmSense, ICASSP 2023, on a Google Soli sensor).
- **Legality is noted, not gated.** Notes assume Turkey (BTK follows CEPT/ETSI).

Evidence tags used below: **CONFIRMED** = text extracted from the vendor datasheet, manual
or product brief; **LIKELY** = vendor web page, distributor page, vendor forum staff post, or
a reputable secondary source; **UNVERIFIED** = only a search snippet, or not found.
Prices are the 2026 single-unit prices seen on the cited page, USD unless marked.

---

## 1. The requirement

### 1.1 Topology A: six-node link, round-robin

Six identical nodes, one transmitter per slot, everyone else listening, exactly the
current ESP32 mesh:

```
t = 1   node 1 TX     nodes 2..6 RX
t = 2   node 2 TX     nodes 1,3..6 RX
...
```

Per node:

- **half duplex**: transmits in its own slot, otherwise receives. No duplexer.
- **two concurrent receive chains** on one LO and one sample clock, each on its own
  antenna, both sampled at the same instant.
- **>= 400 MHz** bandwidth.
- **raw CIR per received frame from both chains**, readable by the host.
- 2-20 GHz preferred.

The two chains are conjugate-multiplied per tap to cancel the carrier and timing offsets
between the transmitting node and the receiving node (tau and psi on deck slide 8). That
only works if both chains share the oscillator phase at the same instant. A part that
switches two antenna ports into one receiver samples them microseconds apart:

```
6.5 GHz carrier, 20 ppm offset  ->  beat = 6.5e9 * 20e-6 = 130 kHz
10 us between the two samples   ->  130e3 * 10e-6 = 1.3 cycles = 470 degrees
```

The product would carry that drift instead of removing it. Switched-port PDoA parts fail
this gate however they are marketed. (The chip's own PDoA output is fine, because the chip
also estimates the offset and corrects it internally; the raw CIR is what is not fine.)

### 1.2 Topology B: six monostatic radars

Each node transmits its own pulse and receives its own echo on the same oscillator. There is
no offset to cancel, so the concurrent-chain gate does not apply. What remains:

- **>= 400 MHz** bandwidth (range bins, see section 2).
- **raw CIR or baseband frames per sweep**, readable by the host.
- a second receive chain is a bonus (angle, two looks at the torso), not a gate.
- 2-20 GHz preferred.

Topology B loses the bistatic and non-line-of-sight geometry that the weapon literature
uses (Yousaf 2025 "non-LOS scattering"), and it loses the link vote across N(N-1) directed
links. It gains: no clock problem at all, and parts that are designed for exactly this use.

### 1.3 Common gates

- **G0 budget**: six nodes, hosts and antennas under USD 3,000. A two-node proof first is
  fine, but the six-node price must fit.
- **G4 raw data**: CIR or IQ, documented or demonstrated, not "ask your module maker".
- **G6 build effort**: attaching antennas, connectors and cables is fine; soldering LGA,
  QFN or BGA parts is not. So every route below is a dev kit, an evaluation board, or a
  USB product, never a bare chip.
- **Host and backhaul**: every node needs a host that talks to the radio (SPI or USB) and
  ships batches to the PC with health, as the ESP32 nodes do now over UDP. Section 8.

---

## 2. Physics: what bandwidth buys

Round-trip range bin `c / (2B)`; one-way delay bin `c / B`. The table is `c / (2B)`.

| Bandwidth B | c/(2B) | Where it comes from |
|---|---|---|
| 20 MHz (ESP32 HT20) | 7.5 m | today |
| 40 MHz (ESP32 HT40) | 3.75 m | today |
| 56-61 MHz (B210, bladeRF) | 2.5 m | SDRs under $3k |
| 160 MHz (Intel AX210 CSI) | 94 cm | widest cheap WiFi CSI |
| 320 MHz (802.11be, no CSI tool yet) | 47 cm | WiFi ceiling |
| 400 MHz (project floor) | 37.5 cm | |
| 460-500 MHz (802.15.4 UWB channel; Novelda X7 RX) | 30-33 cm | SR250, SR150, QM35825, DW3000 |
| 750 MHz (Novelda X7 TX) | 20 cm | |
| 900 MHz (DW1000 receiver, ch 4 or 7) | 16.7 cm | DWM1002 class, if it existed |
| 1.2-1.8 GHz (ARIA Hydrogen, Novelda X4) | 8-12 cm | |
| 4-5.5 GHz (60 GHz radars) | 2.7-3.75 cm | TI, Infineon, Acconeer |
| ~7 GHz (Walabot, 3.3-10.3 GHz) | 2.1 cm | |

Two corrections to rev 1:

1. **The DW1000's 1331 MHz (ch 4) and 1082 MHz (ch 7) are transmit channel widths.** The
   datasheet footnote says "DW1000 maximum receiver bandwidth is approximately 900 MHz" and
   the receiver "can be configured to operate in one of two bandwidth modes; 500 MHz or
   900 MHz" (CONFIRMED, DW1000 datasheet table of channels). The usable CIR bandwidth on a
   DWM1002-class node is therefore ~900 MHz and the bin is 16.7 cm, not 11.3 cm. **Deck
   slides 9 and 13 use the 1331 figure and should be corrected.**
2. **Resolution does not resolve a body-worn object from the torso.** An object 0-5 cm in
   front of the chest is 0-0.3 ns of round-trip delay away from the skin echo; no bandwidth
   on this list separates that. What more bandwidth buys for metal is (a) separating the
   body echo from wall, floor and furniture echoes, so the window statistics look at the
   person and not the room, and (b) many more independent frequency samples for a
   signature, which is the ic27-style inter-carrier feature idea with hundreds of bins
   instead of dozens. For counting it buys (a) and cleaner per-person taps.

---

## 3. Verified facts per part

The evidence table. One row per claim that the ranking depends on. Source URLs are in
section 14; the tag says how the claim was checked.

### 3.1 NXP Trimension SR250 and the SR250-ARD board

| Claim | Evidence | Tag |
|---|---|---|
| Ranging and radar on one chip, 6-8.5 GHz, channels 5 and 9 | NXP product page; Murata Type 2HQ page "UWB Channel 5,9 support" | LIKELY |
| CIR to the host | Fact sheet: "the Trimension SR250 can capture CIR data and provide those to the host for processing" and "Off-chip radar capability, provided by NXP, make it easier to develop use cases that use the host for processing CIRs" | CONFIRMED |
| Product page feature bullet "Radar CIR streaming" | NXP SR250 page | LIKELY |
| Host interface | Fact sheet: presence "reported ... through UWB command interface (UCI)"; board manual: "host interface via SPI and GPIOs" | CONFIRMED |
| Angle | Fact sheet: "Support for 3D and 360-degree AoA, TDOA" | CONFIRMED |
| Board antennas (UM12413, March 2026) | "TRA1 Radar Tx antenna (higher isolation to RXB and RXC)"; "TRA2 Ranging Tx antenna and bottom antenna of the elevation Rx pair"; "RXB Right antenna of the azimuth Rx pair and top antenna of the elevation Rx pair"; "RXC Left antenna of the azimuth Rx pair"; "L-shaped antenna for 3D Angle-of-Arrival"; "tuned for channel 9" | CONFIRMED |
| Number of receivers on the chip | Not public (datasheet UM12508 needs an NXP account). Two RX pairs on the board plus 3D AoA imply at least two receivers; whether both sample in one frame is not stated | UNVERIFIED |
| Public radar demo that records CIRs | GitHub `nxp-uwb/sr250-uwbiot-zephyr`, `uwbiot-top/demos/radar/{demo_radar, demo_ocpd}`; repo docs describe the radar demo capturing "UWB radar Channel Impulse Responses (CIRs)" into `radar_cirs_xxx.csv` | LIKELY (directory CONFIRMED, CSV wording from a search summary of the repo) |
| Host board | Getting-started guide: "FRDM-RW612 Development Board" with a minor rework to route the API pins to the Arduino headers | LIKELY |
| Price and stock | NXP: "$70.00 USD", "Pending Stock". FRDM-RW612: ~$26-50 (Future $26.25, Arrow $29.93, LCSC $49.68) | LIKELY |
| Module route | Murata Type 2HQ (SR250, "3D AoA or 2D AoA", "UWB Radar", "OCPD", SPI): "Under development"; EVK LBUA0ZZ2HQ-EVK by inquiry | LIKELY |
| CIR of a frame received from another node (topology A) | Not found anywhere. The documented CIR path is the radar (own echo) path | UNVERIFIED |

### 3.2 Qorvo QM35825 and the QM35825DK-05 kit

| Claim | Evidence | Tag |
|---|---|---|
| In production; datasheet public | "QM35825 Data Sheet Rev. D, Sep 2025" (LCSC mirror of the Qorvo PDF) | CONFIRMED |
| RF ports | "It integrates UWB LNAs, PA and RF switches, 4 RF ports with highly flexible configuration" | CONFIRMED |
| Two receivers | Datasheet has "Single RX mode" and "Dual RX mode" rows: VDD3 137 mA in dual RX vs 81 mA single RX; sensitivity -101 dBm dual vs -98 dBm single (CH5 Set#03); PDoA accuracy +/-4 deg dual RX vs +/-12.5 deg single RX. Qorvo's Simon Desfarges, forum, 2025-08-11: "There are 2 receivers, combining them (Dual Rx mode), increases the sensitivity of the chip" | CONFIRMED (dual RX mode exists) / LIKELY (two physical receivers) |
| Sampled in one frame | Datasheet: AoA "+/- 2 degree accuracy over a single frame"; also lists "Rx diversity with automatic switching". The 3 dB dual-RX gain and the PDoA improvement only happen if both receivers are on at once. Not stated verbatim | LIKELY |
| CIR | Datasheet: "channel impulse response data computing logic". DK-05 product brief (Rev A, Jul 2025), Radar features: "Channel Impulse Response (CIR) extraction", "Interleaved radar sensing and FiRa ranging", "Presence detection using radar" | CONFIRMED |
| Radar | Datasheet: "ranging, angle of arrival, radar and data transfer functions" | CONFIRMED |
| Host interface | "UCI API" over "industry standard SPI interface"; the DK's PC module is USB | CONFIRMED |
| Kit contents | Brief: "This kit is composed of 2 modules": an RPi module (Raspberry Pi 4, interposer, radio board "integrates 4 RF connectors hosting to the Qorvo Jolie Quad proprietary antenna", preloaded SD card) and a PC module (USB interface board, same radio board, same antenna) | CONFIRMED |
| Price and stock | DigiKey "$693.00", "not stocked at DigiKey; the manufacturer's lead time is 27 weeks"; Qorvo store $599 at qty 1-24 (search snippet) | CONFIRMED (DigiKey) / UNVERIFIED (Qorvo store) |
| Firmware | SDK is a public GitLab repo (`qorvo_sdk/public/devkits/qm35-sdk`) with "Binaries/ - Firmware binaries", Python samples and a licence agreement; Qorvo staff on the forum: "the provided software is a closed source firmware" | LIKELY |
| Wired clock sync between two boards | Qorvo staff: not supported on QM35825 (it was on DW1000/DW3000). Irrelevant for a one-chip two-receiver node; relevant only if someone wanted to gang two chips | LIKELY |
| Sibling parts | QM35825SR "2-antenna version" (DigiKey doc title); QM35725 is the predecessor | UNVERIFIED |

### 3.3 NXP Trimension SR150 and its modules

| Claim | Evidence | Tag |
|---|---|---|
| Dual receiver | Datasheet: AoA "accuracy of +/-3 degrees in one measurement cycle using the on-chip dual receiver architecture"; "SR150 has 2 RX and one TX these can be connected via external switched to antenna matrix"; active states "Idle, TX, RX and Dual RX" | CONFIRMED |
| TX shares an antenna with RX2 | Murata Type 2BP datasheet: "ANT0 ... TX_OUT / RX2_IN ... Tx / Rx switched by internal SPDT"; "ANT1 / ANT2 switched by internal SPDT" into RX1_IN | CONFIRMED |
| Bandwidth | Channels 5 and 9, 499.2 MHz (Murata: "UWB Ch 5 & 9") | LIKELY |
| Raw CIR | Not in the datasheet, not on the product page, not in the Type 2BP EVK brief, not on the MobileKnowledge kit page. The NXP community thread that asked is not reachable (HTTP 403). Nothing public says yes | UNVERIFIED, leaning no |
| Cheapest board | Murata Type 2BP EVK LBUA0VG2BP-EVK-P (Type 2BP + QN9090 BLE MCU + USB-UART): LCSC $58.97, listed at DigiKey/Mouser | LIKELY |
| MK UWB Kit (SR150 anchor + SR040 tags) | "1.800 EUR"; no radar or CIR wording on the page | LIKELY |
| Rev 1 errors | AMOTECH ASMOP1CO0A1 is an **SR040** module (single receiver) per its datasheet title; Ezurio's module is "Sera NX040" on **SR040**; Murata LBUA5QJ2AB-828 is **Type 2AB on Qorvo DW3110/3120**, not NXP. Only ASMOP1BO0N1 (no antenna) and Murata Type 2BP are SR150 | LIKELY |

### 3.4 Decawave/Qorvo DW1000 family and the DWM1002

| Claim | Evidence | Tag |
|---|---|---|
| Channel widths | Ch 4: 3993.6 MHz centre, 3328-4659.2 MHz, "1331.2*"; ch 7: 6489.6 MHz, 5980.3-6998.9 MHz, "1081.6*"; footnote "*DW1000 maximum receiver bandwidth is approximately 900 MHz" | CONFIRMED |
| Accumulator | "Accum Len 1016" (real and imaginary per tap) in a Qorvo support note; the user manual (register 0x25, 992 taps at 16 MHz PRF, ~1 ns spacing) could not be fetched from any mirror in this pass | LIKELY (1016) / UNVERIFIED (spacing) |
| DWM1002 for sale | No listing at DigiKey, Mouser, Octopart or Symmetry; Qorvo product page unreachable (HTTP 429). It shipped only in the "Beta PDoA Kit" (DWM1002 node + DWM1003 tag) | UNVERIFIED, effectively not purchasable |
| Rebuild it | Needs two bare DW1000 (QFN) on one 38.4 MHz reference (Dotlic et al., "Angle of Arrival Estimation Using Decawave DW1000 Integrated Circuits"; AnguLoc, DCOSS 2020). DWM1000 modules carry their own crystal and do not expose the reference, so the module route cannot do it. Fails G6 | LIKELY |

### 3.5 Qorvo DW3000 family (DW3220 / QM33120W, DWM3000, DWM3001C)

| Claim | Evidence | Tag |
|---|---|---|
| One receiver, two switched ports | DW3220 is "2 Antenna Ports" (datasheet title); PDoA variants lose ~1 dB TX power "due to insertion loss associated with the internal PDoA switch"; QM33120W has a PDoA_SW pin that toggles "for AoA switching (within a frame)" | UNVERIFIED (search snippets only, but consistent) |
| CIR readable | Qorvo staff on the forum point to "DW3000 User manual Section 4.7.2" for reading the accumulator over SPI | LIKELY |
| Prices | DWM3001CDK $29.50 (Qorvo store); DWM3000 ~$24, DWM3001C ~$49, QM33120WDK1 $500 (DigiKey) | UNVERIFIED (snippets) |

### 3.6 Monostatic radar modules, 2-20 GHz

| Part | Claim | Evidence | Tag |
|---|---|---|---|
| Novelda X7F202 | "Two integrated Tx/Rx transceivers and antennas allowing digital beamforming"; modes "1 TX + 2 RX"; TX centre 7875 MHz; "TX bandwidth (-10 dB) 750 MHz"; "RX bandwidth (-3dB) 460 MHz"; 2100 MS/s; 192-bin frames; SPI | datasheet (DigiKey mirror) | CONFIRMED |
| Novelda X7 Radar Direct | "two 7.875 GHz transmitters", "two direct RF-sampling receivers", "raw baseband or pre-processed data", "C++ & Python API", USB; kit by request form, no price; the X7F202 dev kit is "no longer available at DigiKey" | novelda.com, DigiKey | LIKELY / CONFIRMED (DigiKey) |
| ARIA Sensing Hydrogen v1p1 SoC | "up to 4 Transmitting and up to 4 Receiving antennas"; "programmable pulse bandwidths (from 500MHz to more than 1.8GHz)"; "7.3GHz to 9.8GHz"; UART/SPI | ariasensing.com | LIKELY |
| ARIA AHM2D / AHM3D / AHM3DSC / LT103 | "Freq. range 7.3GHz to 9GHz", "Interface UART/SPI"; per-module RX count, raw-data API and price not published ("Contact us") | ariasensing.com | LIKELY / UNVERIFIED |
| Vayyar Walabot Developer Pack | "$599.95 USD", "Add to cart" (not retired, contrary to rev 1); "18 antenna array"; "Imaging API, Radar API, and Breathing API", "C++/Python for windows, C++/Python for Linux"; API `Walabot_GetSignal(int txAntenna, int rxAntenna, double **signal, double **timeAxis, int *numSamples)` returns the raw time-domain signal per antenna pair; 3.3-10.3 GHz (US) per the 2015 teardown; pairs are switched, so topology B only | walabot.com, api.walabot.com | LIKELY |
| TDSR P452 | One receiver; two SMA ports through a transfer switch: "For radar use, TX is locked to Port A and we hard-select RX to port B"; multistatic sync is wireless ("no need for a hardware connection between the two units"); 4.0-4.6 GHz or "Full-bandwidth UWB version: 3.1 to 5.3 GHz"; raw scans logged; price not published | datasheet rev H | CONFIRMED / price UNVERIFIED |
| Umain HST-C1R / HST-D3 kit | Single-chip UWB impulse radar, Raspberry Pi 3 kit streams "Radar Raw Data" to a PC; 4.27 GHz, 1-1.7 GHz BW per the 2024 survey; one RX | umain / survey | LIKELY |
| All commercial UWB radar modules in the 2024 survey table | "either two-antenna modules (Novelda, Aria, Humatics, Umain) in a monostatic radar architecture, with one TX and one RX antenna ... or single-antenna modules" | Cheraghinia et al. 2024, arXiv 2402.05649, Table XI | CONFIRMED (survey text) |

### 3.7 60 GHz radars (secondary)

| Part | Claim | Evidence | Tag |
|---|---|---|---|
| Infineon BGT60TR13C, DEMO BGT60TR13C | 1 TX, 3 RX antenna-in-package; 58-63.5 GHz, ">5 GHz bandwidth"; raw ADC frames to a PC over USB through the Radar SDK (`ifxradarsdk` Python wheel); DigiKey "$231.26", 0 in stock | Infineon community, DigiKey | LIKELY / CONFIRMED (price) |
| Infineon BGT60ATR24C | 2 TX / 4 RX, 58-62 GHz, automotive | datasheet title | LIKELY |
| TI IWR6843ISK | 60-64 GHz, "4 receive (RX) 3 transmit (TX)", 4 GHz chirp; raw ADC only through the DCA1000EVM ("enables access to sensor's raw data via LVDS"); ~$283 (DigiKey) | ti.com, DigiKey | CONFIRMED / LIKELY (price) |
| TI IWRL6432BOOST | 57-64 GHz, "Three receive (3RX) two transmit (2TX)"; DCA1000 supported; ~$150 | ti.com | CONFIRMED / LIKELY (price) |
| TI DCA1000EVM | raw-data capture card, ~$719 (DigiKey), one per node | DigiKey | LIKELY |
| TI IWR1843BOOST / AWR1843BOOST | 76-81 GHz, 3 TX 4 RX, 4 GHz; ~$340-405 | ti.com, Mouser | LIKELY |
| Acconeer A121 / XM125 | 60 GHz, 4 GHz BW, pulsed coherent, single antenna pair; XM125 $22-50 | Mouser, SparkFun | LIKELY |
| mmSense (ICASSP 2023) | concealed weapons "with a Miniature Radar Sensor" = Google Soli, 60 GHz | arXiv 2302.14625 | CONFIRMED |

### 3.8 SDRs, RFSoC and WiFi CSI cards (reference)

| Part | Claim | Tag |
|---|---|---|
| Ettus USRP B210 | "$2,387.00", 2 TX 2 RX, "up to 56 MHz of real-time bandwidth" | CONFIRMED |
| Nuand bladeRF 2.0 micro xA9 | "$860.00", 2x2 MIMO, 61.44 MHz | CONFIRMED |
| Lime LimeSDR XTRX | "$960", 2x2, "Bandwidth: 120 MHz" | CONFIRMED |
| Ettus USRP X410 | "$33,020.00", 4 RX 4 TX, "Up to 400 MHz of instantaneous bandwidth per channel" | CONFIRMED |
| Real Digital RFSoC 4x2 | "Academic $2,499.00", AMD University Program end-use form required, "four 5 GSPS ADCs with 6 GHz RF input bandwidth", no RF front end on board | CONFIRMED |
| Any SDR under $3,000 with >= 2 coherent RX and >= 400 MHz | none found; best near-miss is the X410 at 11x the budget | LIKELY |
| Intel AX210 + PicoScenes | "up to 160 MHz bandwidth", packet injection supported, ~$30 card | CONFIRMED |
| Intel BE200 (WiFi 7, 320 MHz) | not in PicoScenes' supported NIC list; 320 MHz only via PicoScenes' SDR baseband | CONFIRMED (absence) |
| Raspberry Pi + Nexmon CSI | "up to 80 MHz bandwidth" | CONFIRMED |
| Espressif esp-crab | two ESP32-C5; page title "Co-Crystal Oscillator CSI Reception"; "single-transmission, dual-reception" mode; bandwidth and sync method not stated | LIKELY / UNVERIFIED |

### 3.9 Parts that are not yet buyable

| Part | Claim | Tag |
|---|---|---|
| ST ST64UWB-A500 / C100 | "Radar sensing (A500 only)"; "1.3 GHz bandwidth of UWB channel 11"; "2x transmit/receive antenna ports"; "currently sampling ... no official pricing has been announced yet" (cnx-software, 2026-03). No eval board found. Simultaneous vs switched: not stated | LIKELY |
| NXP SR200 (mobile), SR048, NCJ29D6 / NCJ29D8 (automotive) | SR200 "Single IC with UWB ranging and radar"; NCJ29D6 "Dual antenna interface for antenna diversity, maximum ratio combining, and angle-of-arrival estimation" plus radar and child-presence detection; none has a public dev board for small quantity | LIKELY |

---

## 4. Gates, applied

- **G0** six nodes with hosts under $3,000.
- **G1 (topology A only)** two concurrent, clock-shared receive chains.
- **G2 (A)** can transmit in its slot. **(B)** has its own transmitter.
- **G3** bandwidth >= 400 MHz.
- **G4** raw CIR or IQ, documented or demonstrated.
- **G5 (soft)** 2-20 GHz.
- **G6** no board-level assembly.

---

## 5. Ranking

### 5A. Topology A: six-node link with conjugate multiply

| # | Part (route) | BW | RX chains | Raw CIR on a received frame | 6 nodes incl. hosts | Blocker |
|---|---|---|---|---|---|---|
| A1 | **Qorvo QM35825**, via QM35825DK-05 (2 radios per kit) | 500 MHz | 2, dual RX mode (CONFIRMED), same frame (LIKELY) | "CIR extraction" on the kit (CONFIRMED) for radar frames; on FiRa/ranging frames from another node: to test | 3 kits $1,797-2,079 + 3 extra Pi hosts ~$200 = **~$2,000-2,300** | 27-week lead time at DigiKey; Qorvo store stock; closed firmware |
| A2 | **NXP SR250**, via SR250-ARD + FRDM-RW612 | 500 MHz | >= 2 (LIKELY from the board's two RX pairs and 3D AoA) | CIR to host is documented for radar (CONFIRMED); for a frame received from another node: to test | 6 x ($70 + ~$30) = **~$600** | "Pending Stock"; receiver count and ranging-CIR not public |
| A3 | **NXP SR150**, via Murata Type 2BP EVK | 500 MHz | 2 (CONFIRMED) | nothing public says yes | 6 x $59 = **~$360** + hosts | buy only if NXP or a two-board test says yes |
| A4 | DWM1002 (2x DW1000, one crystal) | ~900 MHz RX | 2 (CONFIRMED by design) | accumulator, 1016 taps | not for sale; rebuild fails G6 | reference only |
| A5 | ST ST64UWB-A500 / C100 | 500 MHz, ch 11 1.3 GHz | 2 ports, sampling unknown | unknown | sampling only | watch, 2027 |

Everything else that passes G1 (RFSoC 4x2, USRP X410/X440, Ancortek 1T2R kits, Per Vices,
Epiq, AD9081) fails G0 by 5x to 100x and is in section 13.

### 5B. Topology B: six monostatic radars, 2-20 GHz

| # | Part | Band / BW | TX / RX | Raw data | 6 nodes incl. hosts | Blocker |
|---|---|---|---|---|---|---|
| B1 | **NXP SR250-ARD + FRDM-RW612** | ch 5 or 9, 500 MHz | 1 radar TX + azimuth RX pair + elevation pair | radar CIR to host, public demo writes CSV | **~$600** | pending stock; 500 MHz is the floor of the list |
| B2 | **Qorvo QM35825DK-05** | ch 5 or 9, 500 MHz | 4 ports, 2 RX | "CIR extraction", "Presence detection using radar" | **~$2,000-2,300** | lead time; price |
| B3 | **Vayyar Walabot Developer Pack** | 3.3-10.3 GHz, ~7 GHz | 18 antennas, switched pairs | raw signal per pair via `Walabot_GetSignal` | $600 each: **4 nodes $2,400**, 6 nodes $3,600 (over) | USB to a PC-class host per node; no embedded SPI route |
| B4 | **Novelda X7** (X7F202 module, X7 Radar Direct USB kit) | 7.875 GHz, 750 MHz TX / 460 MHz RX | 2 TX + 2 RX direct-RF-sampling | raw baseband, C++/Python, USB | price by request | ask Novelda; RX -3 dB bandwidth only just clears G3 |
| B5 | **ARIA Sensing AHM2D / AHM3D** (Hydrogen SoC) | 7.3-9 GHz, up to 1.8 GHz | up to 4 TX / 4 RX on the SoC; per module unknown | not documented | price by request | ask ARIA: RX count, raw frames, price |
| B6 | TDSR P452 Radar Dev Kit | 4.0-4.6 GHz (600 MHz) or 3.1-5.3 GHz (2.2 GHz) | 1 TX + 1 RX (ports switched) | raw scans | price by request, historically several $k per unit | out of G0 at six |
| B7 | Umain HST-D3 kit | 4.27 GHz, 1-1.7 GHz | 1 / 1 | raw to PC via Pi 3 | price unknown | ask |
| B8 | Qorvo DWM3001CDK (DW3000) | 500 MHz | 1 RX; cannot hear its own pulse, so a 2-board *link* with one chain | accumulator over SPI | $29.50 each | single chain: the ESP32 situation at 500 MHz. **Stepping stone**, not a candidate |

### 5C. Secondary: 60 GHz radars (accepted as candidates, novelty caveat)

| # | Part | BW | TX / RX | Raw data | 6 nodes | Note |
|---|---|---|---|---|---|---|
| C1 | **Infineon DEMO BGT60TR13C** | 58-63.5 GHz, >5 GHz | 1 / 3, all on one clock | raw ADC over USB, Python SDK | **$1,388** | best spec per dollar on the whole list; out of stock at DigiKey (check Mouser/Infineon); 3 cm bins |
| C2 | Acconeer XM125 (A121) | 60 GHz, 4 GHz | 1 / 1 | sparse IQ | $130-300 | cheapest wideband radar anywhere; single chain |
| C3 | TI IWRL6432BOOST + DCA1000EVM | 57-64 GHz | 2 / 3 | raw ADC via capture card | ~$5,200 | the capture card kills it at six |
| C4 | TI IWR6843ISK + DCA1000EVM | 60-64 GHz, 4 GHz | 3 / 4 | raw ADC via capture card | ~$6,000 | same |
| C5 | Infineon BGT60ATR24C | 58-62 GHz | 2 / 4 | automotive | no small-quantity board found | reference |

Why they are secondary: the deck's argument for this project is a cheap COTS link in the
2-20 GHz band, where the concealed-metal literature is thin. At 60-80 GHz the work exists
(mmSense on Soli; the 70-80 GHz scanners on deck slide 4). If the 2-20 GHz routes fail on
raw-data access, C1 is the fallback that keeps three concurrent chains and 5 GHz of
bandwidth for $231 a node.

### 5D. Reference: WiFi CSI cards with two chains

Not candidates (they fail G3), but the cheapest way to test the conjugate-multiply code
on two real chains before any UWB part arrives:

| Part | BW | Chains | Raw CSI | Price | Note |
|---|---|---|---|---|---|
| Intel AX210 + PicoScenes | 160 MHz | 2x2 on one card | yes, plus injection | ~$30 | 94 cm bins; 6 GHz band |
| ASUS RT-AX86U + AX-CSI | 160 MHz | 4x4 | yes | ~$250 | paper PDF not readable this pass |
| Espressif esp-crab (2x ESP32-C5) | WiFi | 2 | yes | not priced | sync method not stated |
| Raspberry Pi + Nexmon | 80 MHz | 1 | yes | owned | today's 5 GHz node |

---

## 6. Budget, six nodes

| Route | Radio per node | Host per node | Six nodes | Two-node proof |
|---|---|---|---|---|
| A2 / B1 SR250-ARD | $70 | FRDM-RW612 ~$30 (has WiFi 6 on board) | **~$600** | ~$200 |
| A3 SR150 Type 2BP EVK | $59 | Pi Zero 2 W / Pi 4 ~$20-80 | **~$500-800** | ~$170 |
| A1 / B2 QM35825DK-05 | $300-347 (half a kit) | Pi 4 included in half the kits; 3 more Pi 4 ~$200 | **~$2,000-2,300** | one kit $599-693 |
| B3 Walabot | $600 | a PC-class host per node (Pi 4 with the Linux SDK, to confirm) | $3,600, over; four nodes $2,400 | $1,200 |
| C1 BGT60TR13C demo | $231 | USB host, Pi 4 ~$60 | **~$1,750** | ~$580 |
| B8 DWM3001CDK stepping stone | $29.50 | on-board nRF52833 + USB | $180 | $59 |

Antennas: SR250-ARD, QM35825DK and the Walabot ship with theirs. The SR150 EVK and the
DW3000 kit have on-board antennas too. Only the ARIA and Novelda routes need a decision.

---

## 7. Detail on the parts that matter

### NXP Trimension SR250 (SR250-ARD + FRDM-RW612)

New to this doc, and the part that fits both topologies and the budget. A 6-8.5 GHz UWB
IC with FiRa 3.0 ranging and an on-chip radar engine. Two ways to use the radar: on-chip
presence detection reported over a GPIO or UCI, or "off-chip radar processing", where the
chip captures CIRs and hands them to the host. NXP's public Zephyr repository has a radar
demo directory and its documentation describes recording radar CIRs to a CSV, so the raw
path is demonstrated, not just promised.

The SR250-ARD ($70) is an Arduino shield with a dedicated radar TX antenna, a ranging TX
antenna that doubles as the elevation RX, and an azimuth RX pair, tuned to channel 9. The
host is the FRDM-RW612 (~$30), an RW612 WiFi 6 + BLE + 802.15.4 MCU board with a small
rework. That pairing is the whole node: UWB radio, host, and WiFi backhaul, for ~$100.

Open on this part: the receiver count on the die (the UCI spec needs an NXP account, free),
whether both receive pairs are sampled in the same frame, and whether a frame received
from *another* SR250 (topology A) also yields a CIR over UCI. The first two come from the
UM12508 UCI specification; the third is a two-board test.

### Qorvo QM35825 (QM35825DK-05)

Qorvo's current UWB SoC, in production with a public Rev D datasheet. Four RF ports behind
switches, LNAs and a PA, and a "Dual RX mode" that draws almost twice the receive current
and gains ~3 dB of sensitivity and a 3x better PDoA over single-RX mode, which is what two
concurrent receivers look like on a datasheet. Qorvo's engineer states it outright: "There
are 2 receivers". The datasheet also names "channel impulse response data computing
logic" and the dev-kit brief lists "Channel Impulse Response (CIR) extraction" under radar,
with radar interleaved with FiRa ranging.

The DK-05 is two complete radios: one on a Raspberry Pi 4 (included) and one on a USB
interface board for a PC, each with a four-element "Jolie Quad" antenna. So three kits are
six nodes, at $599-693 per kit. The risks are procurement (DigiKey quotes a 27-week lead
time; the Qorvo store showed a handful on order) and a closed firmware behind the UCI API.

For topology A the question is the same as for the SR250: is the CIR of a frame received
from the other module exposed, from both receivers? The brief's "CIR extraction" is listed
under radar. One kit answers it.

### NXP Trimension SR150 (Murata Type 2BP EVK)

Confirmed two receivers ("on-chip dual receiver architecture", "Dual RX" state), one TX,
TX sharing antenna 0 with RX2 through an SPDT, RX1 switchable between antennas 1 and 2 for
3D AoA. The Type 2BP EVK is $59 with a BLE MCU and USB-UART on board. It stays on the list
because it is the cheapest confirmed dual-receiver UWB radio on the market, and drops to
third because nothing public says raw CIR comes out, and the SR250 is NXP's own successor
with exactly that feature.

### Decawave DWM1002 (two DW1000 on one crystal), and the DW1000 numbers

Still the cleanest architecture for topology A, and no longer a product: no distributor
lists it, and it only ever shipped in a "Beta PDoA Kit". Rebuilding it needs two bare
DW1000 chips on one 38.4 MHz reference (Dotlic et al.; AnguLoc), which fails G6.
Two corrected numbers for the deck: the usable bandwidth is the receiver's ~900 MHz, not
the 1331 MHz channel; the accumulator is 1016 complex taps at 64 MHz PRF.

### Qorvo DW3000 family

One receiver, two antenna ports, PDoA by switching within the frame. Fails G1 for
topology A and cannot hear itself for topology B. But a DWM3001CDK costs $29.50, exposes the
accumulator over SPI, and two of them make a 500 MHz single-chain link in an afternoon:
real UWB CIR through the existing windows and heads while the two-chain parts are on
order. That is its only role here.

### Vayyar Walabot Developer Pack

Rev 1 called it retired; walabot.com sells it today for $599.95 with an add-to-cart button.
18 antennas on one SoC, 3.3-10.3 GHz in the US version, C++ and Python APIs on Windows and
Linux, and `Walabot_GetSignal(tx, rx, ...)` that returns the raw time-domain signal for any
antenna pair. Pairs are switched, so it is a topology B part: the widest bandwidth of
anything in the 2-20 GHz band at any price on this list, and the platform of the
through-wall pose literature. Six exceed the budget; four do not. The host per node is a
PC-class USB host.

### Novelda X7 (X7F202)

Two transmitters and two direct-RF-sampling receivers at 7.875 GHz, 750 MHz TX bandwidth,
460 MHz RX bandwidth at -3 dB, 2.1 GS/s, 192-bin frames, raw baseband over USB with C++
and Python. The DigiKey dev kit is gone; Novelda sells the "X7 Radar Direct" kit by request
form with no published price. Two receivers on one clock and raw data make it the best
specified monostatic module in band; the unknowns are price and whether the two receivers
are read in the same frame (implied by "digital beamforming", not stated).

### ARIA Sensing AHM2D / AHM3D (Hydrogen SoC)

The rev 1 "highest-value unknown" is half resolved: the Hydrogen SoC supports up to four
transmit and four receive antennas, pulse bandwidths from 500 MHz to over 1.8 GHz, and
7.3-9.8 GHz. What the AHM2D/AHM3D modules expose (RX count, raw frames, price) is still
"contact us". One email.

### Infineon BGT60TR13C (DEMO BGT60TR13C)

The secondary-list winner. One TX, three RX in the package, over 5 GHz of chirp, raw ADC
frames over USB into a Python SDK, $231 per board. Three receivers on one clock at 3 cm
range bins is more than any 2-20 GHz part here offers, at a third of the QM35825 price.
It is second-tier only because of the band decision. Stock was zero at DigiKey on
2026-09-22.

### TI mmWave EVMs

IWR6843ISK ($283, 3T4R, 60 GHz) and IWRL6432BOOST ($150, 2T3R) are the best-documented
radars on the list, but raw ADC needs the $719 DCA1000EVM per node. Without it you get
the on-chip range profiles, not IQ. Out of G0 at six.

---

## 8. Host and backhaul ("the modem" need)

Slide 14 lists "a modem that links the nodes to the PC for data and health checks, as the
ESP32s do now". Per route:

| Route | Radio to host | Host | Backhaul to PC | Reuses today's node protocol? |
|---|---|---|---|---|
| SR250-ARD | SPI + IRQ over Arduino headers | FRDM-RW612 (Cortex-M33, Zephyr, WiFi 6) | the host's own WiFi | yes: UDP batches + health datagrams port over almost intact |
| QM35825DK RPi module | SPI via interposer | Raspberry Pi 4 (included, Linux, Qorvo SDK) | Pi WiFi or Ethernet | yes, in Python on the Pi |
| QM35825DK PC module | USB-C | any Linux host with the SDK: a Pi 4 should do (to confirm) | Pi WiFi | yes |
| SR150 Type 2BP EVK | UART over USB (QN9090 bridge) | Pi Zero 2 W or Pi 4 | Pi WiFi | yes |
| Walabot | USB | PC or Pi 4 (Linux SDK) | Pi WiFi | yes |
| Novelda X7 Radar Direct | USB | PC | | yes if a Pi runs the API |
| BGT60TR13C demo | USB | Pi 4 with `ifxradarsdk` | Pi WiFi | yes |

Time alignment across nodes stays what it is today: wall-clock only, links meet at the
decision. Nothing here needs cross-node phase coherence.

---

## 9. Antenna note

Topology A: two antennas per node, the same pair for transmit and receive through the
chip's switches. On the QM35825DK the Jolie Quad gives four; on the SR250-ARD the board
gives the radar TX, the ranging TX/elevation RX and the azimuth pair. Buy any extra
antennas from one batch: amplitude or phase mismatch between a node's two antennas sits in
the conjugate product as a fixed bias, and is calibrated out with the empty room.

Topology B: one TX and one or more RX antennas per node, all on the board for every route
above except ARIA and Novelda.

---

## 10. Open questions, in cost order

1. **SR250, free**: read UM12508 (SR250 UCI specification, NXP account) for the receiver
   count, whether both RX pairs sample in one frame, and whether a ranging or data session
   reports CIR per received frame. Also read the `demo_radar` source in
   `nxp-uwb/sr250-uwbiot-zephyr` for the CIR notification format and length.
2. **SR250, ~$200**: two SR250-ARD + two FRDM-RW612. Run `demo_radar` (topology B proof),
   then a two-board session and look for CIR on the receiving side (topology A proof).
3. **QM35825, $599-693**: one DK-05. Confirm CIR extraction on a frame received from the
   other module in dual RX mode, and the CIR length and rate. Check Qorvo store stock first.
4. **Novelda, free**: request the X7 Radar Direct kit price and whether the two receivers
   are sampled in one frame.
5. **ARIA, free**: AHM2D and AHM3D receive-channel count, raw-frame API, module price.
6. **SR150, free then $118**: ask NXP whether the SR150 reports CIR over UCI for both
   receivers; if the answer is anything but a documented yes, two Type 2BP EVKs settle it.
7. **Walabot, free**: confirm the Linux SDK runs on a Raspberry Pi 4 (needed for a
   six-node deployment without six PCs) and that `GetSignal` still exists in the current API.
8. **BGT60TR13C, free**: stock at Mouser or Infineon direct; the fallback route needs six.
9. **Deck fixes**: slide 9 and 13 bandwidth for the DWM1002 (900 MHz, 16.7 cm), slide 13
   "DWM1002 availability unclear" becomes "not for sale", SR150 CIR "through UCI" becomes
   "not public", SR250 and QM35825 move into the dashed box with their evidence, Walabot is
   not retired.

---

## 11. Recommendation

Buy in this order, and let each step decide the next:

1. **Now, ~$260**: two SR250-ARD, two FRDM-RW612, and two DWM3001CDK. The DW3000 pair
   gives real 500 MHz UWB CIR into the existing pipeline within a day and is a stepping
   stone only. The SR250 pair answers questions 1 and 2. If the SR250 exposes CIR on
   received frames from both receivers, topology A is solved at ~$600 for six nodes and the
   search ends. If it exposes CIR only in radar mode, topology B is solved at the same price.
2. **If topology A is wanted and the SR250 does not give ranging CIR, $599-693**: one
   QM35825DK-05. Its two radios are the two-node proof. If it passes, three kits are six
   nodes at ~$2,000-2,300, inside G0, subject to lead time; order early.
3. **In parallel, free**: the Novelda and ARIA emails, and the SR150 question to NXP.
   Either quote could put a 750 MHz-1.8 GHz monostatic radar under the SR250 in bandwidth
   terms at an unknown price.
4. **Fallback**: if no 2-20 GHz part gives raw two-chain data, the Infineon DEMO
   BGT60TR13C is six nodes of three concurrent chains at 5 GHz bandwidth for ~$1,750, at
   the cost of the band argument.

Do not buy six of anything before step 1 or 2 has produced a CIR file on disk.

---

## 12. Regulatory notes (Turkey; note only, not a gate)

BTK's short-range-device regulation states alignment with CEPT, ITU and EU decisions,
which for practical purposes means ETSI limits apply:

- UWB 6.0-8.5 GHz (SR250, SR150, QM35825, DW3000 ch 5/9, Novelda, ARIA): -41.3 dBm/MHz
  mean EIRP with no mitigation (ETSI EN 302 065-1). Every dev kit above is certified for it.
- UWB 3.1-4.8 GHz (DW1000 ch 4, TDSR P452, Walabot's lower half): the same limit but only
  with low-duty-cycle or detect-and-avoid mitigation.
- 57-64 GHz (Infineon, TI, Acconeer): licence-exempt SRD under ETSI EN 305 550.
- 24 GHz: only 24.00-24.25 GHz is ISM; 24.25-27.5 GHz is the licensed 5G band n258. The
  Ancortek PUP sweep (24-26 GHz) is not covered for a fixed indoor sensor.
- An SDR or RFSoC transmitting a custom 500 MHz waveform would need to sit at about -14 dBm
  total to meet -41.3 dBm/MHz; not relevant now that those routes are out of budget.

---

## 13. Parts not needed for this phase

Kept for the record, not deleted. Reason in one line each.

**Out of budget (G0), otherwise fine for topology A**

| Part | Why not now |
|---|---|
| AMD RFSoC 4x2 (Real Digital) | $2,499 academic per board, University Program form, and no RF front end; one board is the budget |
| AMD RFSoC 2x2, ZCU111/208/216, AD9081/9082 MxFE | same class, same reason |
| Ettus USRP X410, X440 | $33,020 each |
| Ettus USRP B210 | $2,387, and 56 MHz |
| Nuand bladeRF 2.0 micro xA9, LimeSDR XTRX | $860 / $960, 61-120 MHz, six of them exceed the budget for 1/4 of the bandwidth floor |
| Ancortek SDR-KIT 580AD2 / 980AD2 (1T/2R, 400 MHz) | quote only, historically tens of $k per kit; site unreachable this pass |
| Per Vices Cyan, Epiq Sidekiq X4/X2/NV100, Vesperix VXSDR-20-160, Anritsu MD8190A, Marconi Orion X610, NewEdge RUX105, Epiq NDR374, CesiumAstro SDR-3104, SDR-1001 | instrument or defence pricing, some with unstated IBW |
| Space / military SDR block (Vulcan, Akash, Rocket Lab, Voyager, Honeywell, Innoflight, IQ Technologies, Satlab, Nxbeam, Silvus, Domo, Aeronix, Pacific Defense, CML) | closed waveforms, no sample access |

**Out of band (above 20 GHz), beyond the 60 GHz secondary list**

| Part | Why not now |
|---|---|
| Ancortek PUP_SOLO24P_T2R2, PUP_DUAL24P_T2R4, PUP_EN24C_T2R4, PUP_SOLO24CP_T2R2_HN/RA/ST | 24-26 GHz, quote only, and the sweep is not inside the 24 GHz ISM band |
| InnoSenT ISYS-4001 | 24 GHz, 100 MHz |
| Infineon BGT24MTR12 / BGT24LTR11 / BGT24ATR11 | 24 GHz, 250 MHz at most |
| Uhnder S80 / S81, Calterah CAL77A4T8R, NXP SAF85XX / SAF86XX / SAF8444 | 76-81 GHz automotive imaging radar; no small-quantity eval path found, and TI/Infineon cover the same question cheaper |
| Vayyar vTrigU / VTRIG-74 (62-69 GHz, 20T20R) | availability and price not found |
| Extreme Waves X/Ku/Ka-band | quote only, everything unstated |

**Single receive chain (fails G1 for topology A; not better than B1-B8 for topology B)**

| Part | Why not now |
|---|---|
| Novelda X4M03 / X4 (6-8.5 GHz, 1.5 GHz) | one chain; modules discontinued; X7 replaces it |
| ARIA LT102 V2, LT103 SPI / OEM / OEM XG / XBT / SR | one chain per the 2024 survey; AHM2D/AHM3D are the multi-RX family members |
| GIT Japan GA00491, GA00417 (7.25-10.25 GHz, 3 GHz) | one chain; no purchase path found |
| Skylab SKU611 (switched PDoA), SKU603, SKU609, SKU620 | one chain |
| AI Thinker BU03, BU04 | one chain (DW3000 class) |
| Qorvo DWM1000, DWM1001C, DWM1004C, DWM3000, DWM3001C, QM33120WDK1 | one chain; DWM3001CDK is kept as the $29.50 stepping stone in 5B |
| Insight SiP ISP3010 (DW1000 + nRF52) | one chain |
| Quectel AU30Q (DW3300Q, automotive) | one chain, automotive channel |
| Umain HST-C1R / HST-D3 | one chain; kept in 5B as B7 pending price |
| TDSR P452 | one receiver, price by quote; kept in 5B as B6 |
| Humatics / Time Domain PulsON P440 | discontinued; P452 is the replacement |
| Acconeer XM125 | one chain; kept in 5C as C2 |

**Wrong chip, corrected from rev 1**

| Part | Correction |
|---|---|
| AMOTECH ASMOP1CO0A1 | SR040 (single receiver), not SR150 |
| AMOTECH ASMOP1CO0R1 | chip not confirmed this pass; treat as SR040-class until AMOTECH says otherwise |
| Ezurio / Laird "NX040" | Sera NX040 on SR040, not SR150 |
| Murata LBUA5QJ2AB-828 | Type 2AB on Qorvo DW3110/3120, not NXP |
| Murata Type 2DK (LBUA0ZZ2HQ) | that part number is the **Type 2HQ (SR250)** module, listed in 5A/5B; Type 2DK is the SR040 tag module inside the MK kit |
| AMOSENSE modules | not researched |

**Not yet buyable**

| Part | Status |
|---|---|
| ST ST64UWB-A500 / A100 / C100 | sampling, no eval board, no price (2026-03) |
| NXP SR200, SR048, NCJ29D6, NCJ29D8 | no small-quantity board |
| Murata Type 2HQ module / EVK (SR250) | "under development"; the SR250-ARD is the way to the same chip today |

---

## 14. Sources

Datasheets and manuals read as text in this pass (PDF extracted locally):
- NXP SR150 datasheet (copy) https://www.zlgmcu.com/data/upload/file/Utilitymcu/SR150.pdf
- Murata Type 2BP datasheet https://www.murata.com/-/media/webrenewal/products/connectivitymodule/ultra-wide-band/nxp/sp-vg2bp-forweb.ashx
- NXP Trimension SR250 fact sheet https://www.nxp.com/docs/en/fact-sheet/1834653-Trimension%20SR250_Fact%20Sheet_Lt_v1nm.pdf
- NXP UM12413 SR250-ARD development board user manual, Rev 1.0, 2026-03-04 https://www.nxp.com/docs/en/user-manual/UM12413.pdf
- Qorvo QM35825 Production Data Sheet Rev. D, Sep 2025 (LCSC mirror) https://datasheet.lcsc.com/datasheet/pdf/7fa27c1cc2276788d932bd69b3f392ee.pdf
- Qorvo QM35825DK-05 Product Brief Rev. A, Jul 2025 (DigiKey mirror) https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/7851/2312_QM35825DK-05.pdf
- DW1000 datasheet (Octopart mirror) https://datasheet.octopart.com/DW1000-I-TR13-Decawave-datasheet-101745595.pdf
- Novelda X7F202 datasheet (DigiKey mirror) https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/6456/X7F202%20Datasheet.PDF
- TDSR P452 Data Sheet / User Guide rev H https://tdsr-uwb.com/wp-content/uploads/2024/12/320-0317H-P452-Data-Sheet-User-Guide.pdf
- Cheraghinia et al., "A Comprehensive Overview on UWB Radar", 2024, Table XI https://arxiv.org/abs/2402.05649

Vendor and distributor pages:
- NXP SR250 https://www.nxp.com/products/SR250
- NXP SR250 development board ($70, pending stock) https://www.nxp.com/design/design-center/development-boards-and-designs/TRIMENSION-SR250
- NXP getting started with the SR250 board https://www.nxp.com/document/guide/getting-started-with-the-trimension-sr250-development-board:GS-TRIMENSION-SR250
- NXP SR250 UWBIOT for Zephyr https://github.com/nxp-uwb/sr250-uwbiot-zephyr (radar demos under uwbiot-top/demos/radar)
- NXP SR250 plug-and-play demo https://github.com/nxp-appcodehub/an-sr250-uwb-plug-and-play-demo
- Murata Type 2HQ (SR250) https://www.murata.com/en-us/products/connectivitymodule/ultra-wide-band/nxp/type2hq
- Murata Type 2BP (SR150) https://www.murata.com/en-us/products/connectivitymodule/ultra-wide-band/nxp/type2bp
- Murata Type 2BP EVK at LCSC https://www.lcsc.com/product-detail/C17217752.html and DigiKey https://www.digikey.com/en/products/detail/murata-electronics/LBUA0VG2BP-EVK-P/16680947
- FRDM-RW612 https://www.nxp.com/design/design-center/development-boards-and-designs/FRDM-RW612 (prices: Future, Arrow, LCSC)
- NXP SR150 https://www.nxp.com/products/SR150
- NXP Trimension portfolio https://www.nxp.com/products/wireless-connectivity/trimension-uwb:UWB-TRIMENSION
- NXP NCJ29D6 https://www.nxp.com/products/NCJ29D6
- NXP community, SR150 raw CIR (HTTP 403 this pass) https://community.nxp.com/t5/Wireless-Connectivity/UWB-SR150-Will-the-CIR-raw-data-be-available/td-p/1666761
- MobileKnowledge MK UWB Kit https://www.themobileknowledge.com/product/mk-uwb-kit-sr150-sr040/
- Ezurio Sera NX040 (SR040) https://www.ezurio.com/wireless-modules/ultra-wideband-modules/sera-nx040-series-uwb-bluetooth-le-nfc-modules
- Qorvo QM35825 https://www.qorvo.com/products/p/QM35825
- Qorvo QM35825DK-05 at DigiKey ($693, 27 weeks) https://www.digikey.com/en/products/detail/qorvo/QM35825DK-05/26742402
- Qorvo store QM35825DK-05 https://store.qorvo.com/products/detail/qm35825dk05-qorvo/859278/
- Qorvo forum, QM35825 pre-release questions (two receivers) https://forum.qorvo.com/t/qm35825-pre-release-questions/23206
- Qorvo forum, QM35 boards clock sync https://forum.qorvo.com/t/synchronous-two-qm358-boards-the-rpi-board-and-the-pc-interface-board/24626
- Qorvo QM35 SDK https://gitlab.com/qorvo_sdk/public/devkits/qm35-sdk
- Qorvo forum, DW3000 CIR registers https://forum.qorvo.com/t/channel-impulse-response-power/11513
- Qorvo forum, DWM1002 antenna spacing https://forum.qorvo.com/t/aoa-board-dwm1002-two-antenna-distance-only-2-0154cm-why-no-half-of-wavelength-2-31cm/7026
- Decawave DWM1002 BSP README https://github.com/Decawave/mynewt-dw1000-core/blob/master/hw/bsp/dwm1002/README.md
- Dotlic et al., AoA estimation using DW1000 ICs https://www.researchgate.net/profile/Igor-Dotlic/publication/322408155
- AnguLoc (DCOSS 2020) https://www2.cs.uh.edu/~gnawali/papers/anguloc-dcoss20.pdf
- Novelda X7 Radar Direct https://novelda.com/x7-radar-direct/
- Novelda X7F202 dev kit at DigiKey (no longer available) https://www.digikey.com/en/products/detail/novelda/X7F202-Development-Kit/25558545
- ARIA Sensing Hydrogen v1p1 https://ariasensing.com/ARIA-products/aria-hydrogen-v1p1-chip/
- ARIA Sensing AHM3D https://ariasensing.com/ARIA-products/aria-ahm3d-radar-module/ , AHM2D https://ariasensing.com/ARIA-products/aria-ahm2d-radar-module/ , all products https://ariasensing.com/all-aria-products/
- Vayyar Walabot Developer Pack ($599.95) https://walabot.com/products/walabot-developer-pack-new
- Walabot API, GetSignal https://api.walabot.com/_walabot_a_p_i_8h.html
- Walabot 2015 teardown (3.3-10.3 GHz) https://www.cnx-software.com/2015/10/08/walabot-mimo-radar-board-can-see-through-walls-thanks-to-vayyar-3d-imaging-sensors/
- TDSR radar dev kit https://tdsr-uwb.com/radar-dev-kit/
- Umain HST-D3 kit https://umain.en.ec21.com/HST-D3-Evaluation-Kit--10741357_10742479.html
- ST ST64UWB (cnx-software, 2026-03-13) https://www.cnx-software.com/2026/03/13/st-st64uwb-cortex-m85-ultra-wideband-soc-supports-ieee-802-15-4z-and-802-15-4ab-uwb-standards-radar-sensing/
- ST ST64UWB-A500 https://www.st.com/en/wireless-connectivity/st64uwb-a500.html
- Infineon BGT60TR13C https://www.infineon.com/part/BGT60TR13C ; DEMO board at DigiKey ($231.26) https://www.digikey.com/en/products/detail/infineon-technologies/DEMOBGT60TR13CTOBO1/16580764 ; Radar SDK Python https://community.infineon.com/t5/Radar-sensor/BGT60TR13C-Python-Wrapper-Raw-Data/td-p/616800
- TI IWR6843ISK https://www.ti.com/tool/IWR6843ISK ; IWRL6432BOOST https://www.ti.com/tool/IWRL6432BOOST ; IWR1843BOOST https://www.ti.com/tool/IWR1843BOOST ; DCA1000EVM https://www.ti.com/tool/DCA1000EVM
- Acconeer XM125 https://www.mouser.com/en/new/acconeer/acconeer-xm125-module
- mmSense (ICASSP 2023, Google Soli) https://arxiv.org/abs/2302.14625
- Ettus USRP B210 https://www.ettus.com/all-products/ub210-kit/ ; X410 https://www.ettus.com/all-products/usrp-x410/
- Nuand bladeRF xA9 https://www.nuand.com/product/bladerf-xa9/ ; LimeSDR XTRX https://www.crowdsupply.com/lime-micro/limesdr-xtrx
- Real Digital RFSoC 4x2 https://www.realdigital.org/hardware/rfsoc-4x2
- PicoScenes https://ps.zpj.io/ and supported hardware https://github.com/wifisensing/PicoScenes-github-pages/blob/main/README.md
- Nexmon CSI https://github.com/seemoo-lab/nexmon_csi
- Espressif esp-crab https://github.com/espressif/esp-csi/blob/master/examples/esp-crab/README.md
- Ancortek https://ancortek.com/ (TLS error this pass); PUP series via everythingRF https://www.everythingrf.com/news/details/7920-A-Low-power-Compact-Software-Defined-Radar-Kit-for-K-Band-Applications-from-24-to-26-GHz
- Uhnder S80 https://www.uhnder.com/images/data/S80_PTB_v2.0_1_.pdf ; S81 https://www.uhnder.com/images/data/S81_PTB_v1.0_(1)_.pdf
- ETSI EN 302 065-1 https://www.etsi.org/deliver/etsi_en/302000_302099/30206501/02.01.01_60/en_30206501v020101p.pdf ; EN 305 550 https://www.etsi.org/deliver/etsi_en/305500_305599/305550/02.01.00_20/en_305550v020100a.pdf ; EU 2019/784 (24.25-27.5 GHz) https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX%3A32019D0784 ; BTK https://www.btk.gov.tr/frekans-tahsisinden-muaf-telsiz-cihaz-ve-sistemleri
- Prior project shortlist ~/Desktop/docProcess/WaveTrace/reports/UWB_RADAR_SHORTLIST.md (superseded by this file)
