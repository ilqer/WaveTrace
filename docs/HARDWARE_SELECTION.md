# Hardware Selection — RF Front End for the 6-Node Mesh

Status: research complete, no purchase made. Written 2026-09-21.
Scope: choosing the radio hardware that replaces the ESP32 mesh for Stage-E
(concealed weapon / metal) and people counting. Candidates were drawn from
everythingrf category listings plus independent research.

This document is a decision record and a parts reference. It is not build truth —
nothing here is implemented.

---

## 1. The requirement

Six identical nodes in a time-division round-robin, matching the existing ESP32
mesh topology (one transmitter per slot, everyone else listening):

```
t = 1   node 1 TX     nodes 2..6 RX
t = 2   node 2 TX     nodes 1,3..6 RX
...
```

Per-node hardware spec:

- **half duplex** — a node either transmits or receives, never both at once.
  No duplexer, no TX/RX isolation problem, no separate TX antenna.
- **must be able to transmit** in its own slot.
- **two concurrent receive chains**, sharing one LO and one sample clock, each on
  its own antenna. Both sampled at the same instant.
- **>= 400 MHz** bandwidth.
- **2-20 GHz** band.
- **raw data out** — CIR accumulator or IQ samples, not a processed target list.
- affordable and assemblable at six units. Attaching an antenna or a connector is
  acceptable; board-level assembly of fine-pitch or BGA parts is not.

### Why the two receive chains must be concurrent, not switched

The two antennas are conjugate-multiplied to cancel clock drift (CFO/SFO), the
same trick the ESP32 pipeline uses. That cancellation only works if both antennas
see the same oscillator phase at the same instant.

A switched two-port receiver samples antenna A and antenna B a few microseconds
apart. At 6.5 GHz with a 20 ppm oscillator offset:

```
beat frequency = 6.5e9 * 20e-6 = 130 kHz
phase walk over a 10 us gap = 130e3 * 10e-6 = 1.3 cycles = 470 degrees
```

The conjugate product would carry that drift instead of removing it. Switched-port
PDoA parts therefore do not satisfy the requirement, however they are marketed.

---

## 2. Physics: what bandwidth buys

Radar range resolution is `c / (2B)`. One-way delay resolution is `c / B`.
A concealed object on a torso is roughly 10-30 cm, so ~1 GHz is where metal
becomes resolvable at all.

| Bandwidth | Range resolution | Comment |
|---|---|---|
| 40 MHz (ESP32 HT40, current) | 7.5 m | presence only |
| 56-61 MHz (B210, bladeRF) | 2.5 m | counting, not metal |
| 100 MHz (24 GHz ISM radar) | 1.5 m | speed guns, presence |
| 250 MHz (24.0-24.25 ISM) | 60 cm | still too coarse |
| 400 MHz (project floor) | 37.5 cm | marginal for a body-worn object |
| 500 MHz (UWB channel) | 30 cm | marginal |
| 600 MHz (TDSR P452) | 25 cm | workable |
| 900 MHz (DW1000 ch4/ch7) | 17 cm | good |
| 1.5-2 GHz (Novelda, ARIA) | 8-10 cm | good |
| 2.5 GHz (Ancortek 24 GHz) | 6 cm | very good |
| 5 GHz (76-81 GHz automotive) | 3 cm | best available |

---

## 3. Selection gates

Applied in order. Failing G1 ends the evaluation regardless of other merits.

- **G1** two concurrent, clock-shared receive chains
- **G2** can transmit
- **G3** bandwidth >= 400 MHz
- **G4** raw data (CIR or IQ) accessible
- **G5** (soft) band within 2-20 GHz
- **G6** (soft) cost and build effort at six units

---

## 4. Ranking

### Group A — passes all four hard gates

| # | Part | Vendor | Band GHz | BW | RX chains | Cost x6 | Blocker |
|---|---|---|---|---|---|---|---|
| 1 | **DWM1002 PDoA node** | Decawave / Qorvo | ch4 3.99 / ch7 6.49 | **1331 / 1082 MHz** | 2x DW1000, one shared crystal | ~$1-2k | availability; likely EOL |
| 2 | **QM35825** (+ QM35825DK-05) | Qorvo | 6.24-8.24 | 500 MHz | 4 RF ports, on-chip radar | ~$1k | port concurrency unconfirmed |
| 3 | RFSoC 4x2 | AMD / Real Digital | DC-6 | GHz-class | 4 ADC, same die | ~$14k | FPGA development |
| 4 | SR150 modules | NXP via AMOTECH / Murata / Ezurio | 6.24-8.24 | 500 MHz | 2 concurrent on-chip | ~$200 | raw CIR unconfirmed |
| 5 | ST64UWB-C100 / A500 / A100 | STMicroelectronics | 6.49-8.99 | 500 MHz | 2 ports | ~$200 | concurrency + raw unconfirmed |
| 6 | USRP X410 | Ettus / NI | 0.001-7.2 | 400 MHz/ch | 4 coherent | ~$150k | price |
| 7 | USRP X440 | Ettus / NI | 0.03-4 | 1.6 GHz (2ch) | 8 coherent | ~$150k | price, 4 GHz ceiling |
| 8 | SDR-KIT 580AD2 | Ancortek | 5.6-6.0 | 400 MHz | 1T / 2R | ~$30-60k | price, no MAC |
| 9 | SDR-KIT 980AD2 | Ancortek | 9.6-10.0 | 400 MHz | 1T / 2R | ~$30-60k | price, no MAC |
| 10 | Cyan | Per Vices | DC-18 | multi-GHz/ch | up to 16 | six figures each | price |
| 11 | Sidekiq X4 | Epiq | ~0.001-6 | ~1 GHz | multi | high | needs host, confirm specs |
| 12 | AD9081/9082 MxFE EVB | Analog Devices | DC-~6 | GHz-class | 4 ADC, same die | ~$42k+ | FPGA carrier per node |

### Group B — out of band, otherwise excellent

| # | Part | Vendor | Band GHz | BW | RX | Note |
|---|---|---|---|---|---|---|
| 13 | PUP_SOLO24P_T2R2 | Ancortek | 23.5-26 | 2.5 GHz | 2T/2R | exact node topology, 6x the in-band BW, band 20% over |
| 14 | PUP_DUAL24P_T2R4 | Ancortek | 23.5-26 | 2.5 GHz | 2T/4R | four coherent RX, onboard patches |
| 15 | PUP_EN24C_T2R4 | Ancortek | 23.5-26 | 2.5 GHz | 2T/4R | six external antennas, 20-22 dBm, aimable |
| 16 | PUP_SOLO24CP_T2R2_HN/_RA/_ST | Ancortek | 23.5-26 | 2.5 GHz | 2T/2R | prebuilt horns for directional TX |
| 17 | S81 | Uhnder | 76-81 | 5 GHz | 8R, 96 MIMO | digital code modulation; vendor ~32 staff Aug 2026 |
| 18 | S80 | Uhnder | 76-81 | 5 GHz | 16R, 22 dBm | more channels, same vendor risk |
| 19 | CAL77A4T8R | Calterah | 76-81 | 5 GHz | 8R, 4T | cascadable, healthier vendor |
| 20 | SAF85XX | NXP | 76-81 | 5 GHz | 4R, 4T | most accessible mmWave eval path |
| 21 | SAF86XX | NXP | 76-81 | 5 GHz | 4R, 4T | adds Ethernet data path |
| 22 | SAF8444 | NXP | 76-81 | 5 GHz | - | power-efficient variant |

### Group C — fails G1 (single receive chain)

| # | Part | Vendor | Band GHz | BW | Note |
|---|---|---|---|---|---|
| 23 | AHM3D / AHM3DSC | ARIA Sensing | 7.3-9 | 1.7 GHz | MAY PASS G1 - reports angle, implying multiple RX. VERIFY |
| 24 | AHM2D / AHM2DSC | ARIA Sensing | 7.3-9 | 1.7 GHz | same uncertainty, 2D, 12 m |
| 25 | P452 Radar Dev Kit | TDSR | 4.0-4.6 | 600 MHz | was #1 under old spec; multistatic is cross-node not cross-antenna |
| 26 | GA00491 | GIT Japan | 7.25-10.25 | 3 GHz | widest in-band BW on the list, single chain |
| 27 | X4M03 XeThru | Novelda | 6.0-8.5 | 1.5 GHz | best raw-data ergonomics, wrong topology |
| 28 | LT102 V2 | ARIA Sensing | 6.5-8.5 | 2 GHz | presence/distance only |
| 29 | LT103 SPI/OEM/OEM XG/XBT/SR | ARIA Sensing | 7.3-8.5 | 1.2 GHz | SPI variant best odds of raw access |
| 30 | GA00417 | GIT Japan | 7.25-10.25 | 3 GHz | superseded by GA00491 |
| 31 | SKU611 | Skylab | 6.24-6.739 | 500 MHz | PDoA but SWITCHED, not concurrent - fails per section 1 |
| 32 | DWM1000 | Qorvo | 3.244-6.999 | up to 900 MHz | reference part for published CIR research |
| 33 | SKU603 | Skylab | 3.244-6.999 | up to 900 MHz | reaches DW1000 channel 4 |
| 34 | DWM3001C | Qorvo | 6.5-8 | 500 MHz | most complete Qorvo module, DWM3001CDK |
| 35 | DWM3000 | Qorvo | 6.25-8.25 | 500 MHz | DWM3000EVB Arduino shield form |
| 36 | BU04 | AI Thinker | 6.49 / 7.99 | 500 MHz | cheap, available, onboard MCU |
| 37 | SKU620 | Skylab | 6.24-6.739 | 500 MHz | USB interface, rare in this class |
| 38 | BU03 | AI Thinker | 6.49 / 7.99 | 500 MHz | SPI/GPIO only, cheaper |
| 39 | SKU609 | Skylab | 3.774-4.243 + 6.24-6.739 | 500 MHz | dual band |
| 40 | DWM1001C | Qorvo | 6.24-6.739 | 500 MHz | DW1000 + nRF52 + BLE |
| 41 | DWM1004C | Qorvo | ch2 + ch5 | 500 MHz | DW1000 + MCU + motion sensor |
| 42 | ISP3010 | Insight SiP | 4.243-4.742 | 500 MHz | self-contained, narrowest coverage |
| 43 | AU30Q | Quectel | 6.5-8 | 500 MHz | automotive grade |

### Group D — fails G3 (bandwidth), topology fine

| # | Part | Vendor | Band GHz | BW | RX | Cost x6 | Note |
|---|---|---|---|---|---|---|---|
| 44 | B210 | Ettus / NI | 0.07-6 | 56 MHz | 2 concurrent | ~$9k | passes G1/G2/G4 cleanly; right prototype, wrong product |
| 45 | bladeRF 2.0 micro xA9 | Nuand | 0.047-6 | 61 MHz | 2x2 MIMO | ~$4.3k | same trade, cheaper, smaller community |
| 46 | VXSDR-20-160 | Vesperix | 5-20 | ~128 MHz | unstated | - | best tuning range, 3x short on BW |
| 47 | MD8190A | Anritsu | 0.4-6 | <=200 MHz est | - | - | best docs and purchase path of the SDRs |
| 48 | Orion X610 | Marconi | 2.2-5 | unstated | 1 or 4 (listing conflicts) | - | monitoring SDR, likely exposes IQ |
| 49 | RUX105 | NewEdge | 3-6 | unstated | - | - | VPX chassis x6 impractical |
| 50 | ISYS-4001 | InnoSenT | 24.115-24.215 | 100 MHz | 8R | - | out of band and 4x short on BW |

### Group E — fails multiple gates, or unobtainable

| # | Part | Vendor | Why |
|---|---|---|---|
| 51 | Type 2DK / LBUA0ZZ2HQ / LBUA0VG2BP / LBUA5QJ2AB-828 | Murata | SR-family silicon, dual-RX unconfirmed per part. CHECK INDIVIDUALLY - some may belong in Group A |
| 52 | NDR374 | Epiq | 4 channels, IBW unstated, VPX chassis x6 impractical |
| 53 | SDR-3104 | CesiumAstro | 4 ch, Rx 0.1-31 GHz, IBW unstated, space pricing |
| 54 | SDR-1001 | CesiumAstro | 8 ch, 0.3-6 GHz, IBW unstated, space pricing |
| 55 | Walabot Developer | Vayyar | 3.3-10.3 GHz, 18 antennas on one SoC, ~7 GHz BW, USB, Python/MATLAB. Would have ranked #1. RETIRED |
| 56 | PulsON P440 | Humatics / Time Domain | 3.1-4.8 GHz, mono/bi/multistatic. DISCONTINUED over 5G interference, replaced by P452 |
| 57 | X/Ku/Ka-band | Extreme Waves | 8-31 GHz, integrated array, FMCW, angle. In band; bandwidth, interface and price all unstated. Quote-only |
| 58 | Space / military SDR block (19 products) | Vulcan Wireless, Akash, Rocket Lab, Voyager, Honeywell, Innoflight, IQ Technologies, Satlab, Nxbeam, Silvus, Domo, Aeronix, Pacific Defense, CML | satellite TT&C terminals and tactical MANET radios. Fixed bands, closed waveforms, no sample access, defence pricing |

---

## 5. Detail on the parts that matter

### 3. AMD RFSoC 4x2 (Real Digital)

Zynq UltraScale+ RFSoC ZU48DR, Gen 3. Four 14-bit ADCs at 5 GSPS with 6 GHz
analog input bandwidth; two 14-bit DACs at 9.85 GSPS. Converters, ARM cores and
FPGA fabric are on one monolithic die, which gives the tightest cross-channel
timing available anywhere on this list.

Maps onto the node spec with headroom: two ADCs are the coherent receive pair,
two spare, a DAC is the transmit slot. Bandwidth is ~15x the requirement.
Academic pricing is modest; six units land around $14k.

The cost is development. This is Vivado, HDL, DMA plumbing and a hand-written
round-robin MAC, six times over. WaveTrace's codebase is Python and C++; this is
a different discipline and realistically a semester before first usable data.

Verdict: the only candidate satisfying every technical gate with no open
questions. Blocked only by effort.

### 4. NXP Trimension SR150 (modules: AMOTECH ASMOP1CO0A1 / ASMOP1BO0N1 / ASMOP1CO0R1, AMOSENSE, Murata LBUA, Ezurio NX040)

On-chip dual receiver architecture. NXP states AoA accuracy of better than
+/-3 degrees measured "in one measurement cycle" - concurrent, not switched,
which is what the conjugate multiply requires. Three UWB RF ports for 2D/3D AoA.
On-board CoolFlux BSP32 DSP running ToF, AoA and, per NXP's own description,
radar algorithms. IEEE 802.15.4z, 500 MHz channels, 6.24-8.24 GHz.

Full transceiver, so every node can transmit. 802.15.4z is natively half-duplex
and slotted, which means the existing round-robin token-passing design ports
across almost intact - the largest software saving of any candidate.

Modules run roughly $20-50 each, so six nodes cost a couple of hundred dollars.
ASMOP1CO0A1 ships with an antenna; ASMOP1BO0N1 and ASMOP1CO0R1 do not.

THE RISK: raw CIR access. Host communication is FiRa-standard UCI over SPI, and
raw CIR is not a documented UCI capability. An NXP community thread asking
exactly "will the CIR raw data be available?" was marked solved by redirecting
the asker to their module maker. That is a soft no, not a yes.

Verdict: perfect topology at negligible cost, gated entirely on one unanswered
question. Resolve before buying six.

### 5. ST ST64UWB-C100 / A500 / A100

New ST UWB SoC family. Channels 5, 6, 8, 9, 10, 12 (6.49-8.99 GHz), BPRF and
HPRF, dual transmit and receive antenna ports. Supports SS-TWR, DS-TWR, AoA and
PDoA. Also claims 802.15.4ab, the amendment that standardises UWB sensing modes
rather than only ranging, plus narrowband assistance at 5.7-6.4 GHz.
A500 and A100 are automotive-qualified; C100 is the industrial/consumer part and
the easier one to buy in small quantity.

Two RX ports are confirmed; whether they sample concurrently is not. Raw CIR
access is not documented either way. Newer silicon may be more open than NXP's,
or less.

Verdict: same shape as the SR150 with one more unknown. Ask ST the same question.

### 6/7. Ettus USRP X410 and X440

X410: 1 MHz to 7.2 GHz (tunable to 8), 400 MHz instantaneous bandwidth on each of
4 TX and 4 RX channels, two-stage superheterodyne, phase-coherent operation via
built-in GPSDO or external 10 MHz + 1 PPS. Half-rack 1U.

X440: 30 MHz to 4 GHz, direct sampling at up to 2 GSPS per channel giving
1.6 GHz instantaneous bandwidth on 2 channels, or 400 MHz across all 8.
8 TX / 8 RX, phase coherency by shared sample clock, balun-coupled MMPX front end.

Both satisfy every gate. X410 additionally covers the 5 GHz WiFi band, so it
could run the existing CSI work in the same box. Neither ships radar software -
the waveform and processing are yours to write.

Verdict: architecturally flawless, roughly $150k for six. Out of reach.

### 8/9. Ancortek SDR-KIT 580AD2 and 980AD2

580AD2: C-band, centre frequency and bandwidth selectable within 5.6-6.0 GHz,
one transmitter and two receiver channels, ~110 m on 1 cm^2 RCS. Vendor names
through-the-wall, occupancy, gesture and human activity monitoring.

980AD2: X-band, 9.6-10.0 GHz, one transmitter and two receiver channels,
19 dBm typical output, ~80 m on 1 cm^2 RCS. Explicitly designed to support
interferometric radar and direction-of-arrival measurement.

Both: SDR-GUI over USB 2.0 selects waveform, centre frequency, bandwidth,
sampling rate, filtering and display parameters, and records and exports raw I/Q.
The receive mixer output is digitised and streamed to the host in real time.

1T/2R is literally the node spec, and the raw I/Q path is exactly what Stage-E
needs. The problems are that these are standalone radars with no networking
layer, no shared timebase across units and no TX/RX role-switching protocol, and
that six kits cost roughly $30-60k.

Verdict: right per-node topology, wrong economics and no mesh layer. Viable at
two units to prove the physics before funding six.

### 13-16. Ancortek PUP series (24 GHz)

23.5-26 GHz, so a 2.5 GHz span - six times the bandwidth of the in-band Ancortek
kits, giving ~6 cm resolution. Plug-and-play USB with the same acquisition
software. FMCW and CW.

- PUP_SOLO24P_T2R2: 2 Tx / 2 Rx, 4 onboard patch antennas, 16-18 dBm.
  Exact node topology.
- PUP_DUAL24P_T2R4: 2 Tx / 4 Rx, 6 onboard patches, 16-18 dBm.
- PUP_EN24C_T2R4: 2 Tx / 4 Rx, 6 external antennas, 20-22 dBm.
- PUP_SOLO24CP_T2R2_HN / _RA / _ST: 2 Tx / 2 Rx, 2 patch + 2 external antennas;
  suffix denotes external antenna type (horn and others).

Band is 20% above the stated 2-20 GHz ceiling. If that ceiling is a preference
rather than a regulatory constraint, these are the strongest capability jump
available: same vendor, same software, same topology, six times the bandwidth.

OPEN: confirm whether 24 GHz ISM is open for a fixed indoor installation in the
deployment jurisdiction. In most regions it is.

### 23. ARIA Sensing AHM3D / AHM3DSC — HIGHEST-VALUE UNKNOWN

7.3-9 GHz, so ~1.7 GHz of bandwidth. Embedded antennas. Listed measurements
include movement, distance, ANGLE, direction, presence and speed. Angle implies
more than one receive chain.

If those chains are concurrent and the module exposes raw frames, this jumps
straight to the top of Group A: in-band, 1.7 GHz (3.4x the SR150), module form
factor, and far cheaper than any SDR. It would beat the SR150 on every axis.

Cheapest possible upside on the whole list. Email ARIA Sensing.

### 25. TDSR P452

4.0-4.6 GHz, 600 MHz RF bandwidth at 4.3 GHz centre. Fully coherent monostatic,
bistatic and multistatic radar modes. 2 cm ranging accuracy at up to 125 Hz.
Maximum range 300-1100 m depending on antenna height and ground. Redesigned RF
section filters out 5G interference. FCC Part 15 and ETSI EN 302 065 compliant.
Built on TDSR's FIFE (Fully Integrated Front End) custom silicon. Sold as a
Radar Development Kit with MRM/CAT software; datasheet rev H, December 2024.
SMA connectors, so antennas screw on.

Ranked #1 before the node spec was pinned down. It fails G1: one receive chain.
Its multistatic mode coordinates across separate modules, which is cross-node
diversity, not the two clock-shared antennas the conjugate multiply needs.

Still the best single-RX UWB radio found, and the replacement for the
discontinued P440.

### 27. Novelda X4M03 (XeThru)

X4 SoC, 6.0-8.5 GHz, ~1.5 GHz bandwidth, impulse radar with on-chip sampler.
Three interconnecting boards: X4SIP02 radar subsystem, X4A02 antenna board,
XTMCU02 MCU board. XeThru Module Connector is a DLL/shared object for
Windows, Linux and macOS with MATLAB, Python and C++/C bindings, streaming raw
frames over USB. Sub-mm stated accuracy, simultaneous observation range to 10 m.

Best raw-data ergonomics of any UWB radar found. Single receive chain, and two
X4M03 units do not share a clock, so it fails G1 outright.

### 44/45. USRP B210 and bladeRF 2.0 micro xA9 — the prototype path

B210: 70 MHz to 6 GHz, 2 TX and 2 RX genuinely simultaneous and phase-coherent,
56 MHz bandwidth, fully open through UHD and GNU Radio, ~$1.5k each.
bladeRF 2.0 micro xA9: 47 MHz to 6 GHz, 2x2 MIMO, 61.44 MHz, ~$720 each.

Both pass G1, G2 and G4 cleanly and fail G3 by roughly 7x. Round-robin is a
weekend of work on either. Six B210s is ~$9k; six bladeRFs ~$4.3k.

At 56 MHz the resolution is 2.5 m, so metal detection is off the table. These are
the fastest route to a *working* six-node coherent mesh, proving the software,
the MAC, the conjugate multiply and the counting model - and then being replaced.

### 55. Vayyar Walabot Developer — RETIRED

3.3-10.3 GHz in the US band (6.3-8 GHz in the EU variant), VYYR2401-A3 IC,
18 antennas on one SoC so coherence is inherent, micro-USB 2.0 for data and
power, Python and MATLAB APIs, vendor library of imaging algorithms.
~7 GHz of bandwidth. A published body of through-wall pose-imaging research
uses exactly this board.

Would have ranked #1 on every criterion. Core Electronics keeps the listing
"for reference purposes only"; current Walabot products are consumer stud
finders with no raw-data API. Second-hand only.

---

## 6. Open questions, in cost order

1. **ARIA Sensing AHM3D** - are the receive chains concurrent, and does the
   module expose raw frames? Free. Highest upside on the list. (#23)
2. **NXP SR150 raw CIR** - does the SR150 expose per-frame raw CIR accumulator
   samples from BOTH receivers via a vendor-specific UCI command, and is that
   under standard NDA or module-maker-only? Ask NXP and the module maker; also
   settle empirically by buying two modules (~$100). (#4)
3. **ST ST64UWB** - same two questions: concurrent RX, and raw CIR access. (#5)
4. **24 GHz regulatory** - is 23.5-26 GHz open for a fixed indoor installation in
   the deployment jurisdiction? If yes, #11 changes the project. (#13-16)
5. **Ancortek pricing** - actual quote for 580AD2 and 980AD2 at 1, 2 and 6 units.
6. **Murata LBUA / Type 2DK** - which of these are SR150-based with dual
   concurrent RX? Some may belong in Group A. (#51)

## 7. Antenna note

Half duplex means **two antennas per node, not three** - the same pair serves
transmit and receive through the chip's internal switch.

Buy all twelve from one batch. Any amplitude or phase mismatch between a node's
two antennas appears directly in the conjugate product as a fixed bias.

## 8. Recommendation

The ranking has a clean shape: #1 RFSoC and #2 SR150 are the only real
candidates, and they fail in opposite directions - one costs a year of FPGA work,
the other may not expose the data at all. Everything between them is out of band,
out of budget, or single-RX.

De-risk in the order that costs least: question 1, then question 2, then decide.
Do not buy six of anything before question 2 is answered.

## 9. Sources

- Qorvo DW3220 https://www.qorvo.com/products/p/DW3220
- Qorvo DW1000 user manual https://www.qorvo.com/products/d/da007967
- Qorvo MDEK1001 EOL notice https://www.qorvo.com/products/p/MDEK1001
- NXP Trimension SR150 https://www.nxp.com/products/SR150
- NXP SR150/SR040 fact sheet https://www.nxp.com/docs/en/fact-sheet/UWBIOTFSA4.pdf
- NXP SR150 datasheet https://www.zlgmcu.com/data/upload/file/Utilitymcu/SR150.pdf
- NXP community, SR150 raw CIR https://community.nxp.com/t5/Wireless-Connectivity/UWB-SR150-Will-the-CIR-raw-data-be-available/td-p/1666761
- NXP ASMOP1BO0N1 SR150 module https://www.nxp.com/products/security-and-authentication/authentication/trimension-sr150-module-without-antennas-asmop1bo0n1:ASMOP1BO0N1
- ST ST64UWB-A500 https://www.st.com/en/wireless-connectivity/st64uwb-a500.html
- ST ST64UWB-C series flyer https://www.st.com/resource/en/flyer/stmicroelectronics-st64uwb-c-series-flyer.pdf
- Ancortek SDR-KIT 580AD2 https://ancortek.com/sdr-kit-580ad2
- Ancortek RF modules overview http://ancortek.com/rf-modules-overview
- Ettus USRP X410 https://www.ettus.com/all-products/usrp-x410/
- Ettus USRP X440 knowledge base https://kb.ettus.com/X440
- AMD RFSoC 4x2 https://www.realdigital.org/hardware/rfsoc-4x2
- Novelda X4M03 https://www.radartutorial.eu/19.kartei/13.labs/karte009.en.html
- Novelda products https://novelda.com/products/
- TDSR P452 datasheet https://tdsr-uwb.com/wp-content/uploads/2024/12/320-0317H-P452-Data-Sheet-User-Guide.pdf
- TDSR P452 radar dev kit https://tdsr-uwb.com/radar-dev-kit/
- TDSR about, P440 discontinuation https://tdsr-uwb.com/about/
- Vayyar Walabot teardown/specs https://www.cnx-software.com/2015/10/08/walabot-mimo-radar-board-can-see-through-walls-thanks-to-vayyar-3d-imaging-sensors/
- Uhnder S80 product brief https://www.uhnder.com/images/data/S80_PTB_v2.0_1_.pdf
- Uhnder S81 product brief https://www.uhnder.com/images/data/S81_PTB_v1.0_(1)_.pdf
