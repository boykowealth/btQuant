# SIP Global — Commodity Market Cluster Taxonomy

*Clusters are defined by pricing mechanism, not commodity type. Markets in the same cluster share the same quantitative framework, index design logic, and trading infrastructure.*

---

## Cluster A — Seasonal Agricultural Bulk

Harvest-driven mean reversion with strong grade and transport basis. The dominant SIP beachhead — hay, stover, and forage markets globally follow the same OU + seasonality framework.

| Characteristic | Description |
|---|---|
| Price driver | Annual harvest cycle + weather |
| Grade basis | Moisture, protein, test weight |
| Price frequency | Weekly to monthly observations |
| Transport | 30–60% of value — geography is critical |
| Model class | OU, STL seasonality, grade adj, local forward curve |

| Market | Region |
|---|---|
| Hay (timothy, alfalfa) | Alberta, NZ, Australia, Argentina |
| Corn stover | US Midwest |
| Ryegrass silage | UK, Ireland |
| Wheat straw | India, France |
| Cover crop biomass | US Corn Belt |

---

## Cluster B — Weather-Event Demand

Demand is triggered by discrete weather events, not consumption schedules. Price structure is pre-season procurement vs in-season emergency spot — the spread between them is the trade.

| Characteristic | Description |
|---|---|
| Price driver | Snowfall / temperature / drought events |
| Demand structure | Poisson jumps, regime switching |
| Buyer type | Municipalities, DOTs, airports |
| Storage | Indefinite — constraint is logistics timing |
| Model class | Compound Poisson, Markov switching, barrier options |

| Market | Region |
|---|---|
| Road salt / rock salt | North America, Northern Europe |
| Calcium / mag chloride | Canada, US, Scandinavia |
| Sand & abrasives | Canada, US Midwest |
| Dust suppressants | Mining regions globally |
| Emergency water supply | Drought-prone regions |

---

## Cluster C — Specialty Food & Beverage

Origin and certification premiums dominate over supply/demand fundamentals. Grade stratification is the primary source of price complexity — blending grades destroys the index signal.

| Characteristic | Description |
|---|---|
| Price driver | Origin, certification, quality tier |
| Grade basis | Certification premium (organic, fair trade, ceremonial) |
| Buyer type | Food manufacturers, restaurants, supplement brands |
| Transport | Low — high value density absorbs freight |
| Model class | Quality premium model, grade surface, basis vs commodity proxy |

| Market | Region |
|---|---|
| Matcha | Japan, China |
| Single-origin spices | India, Indonesia, Vietnam, Guatemala |
| Specialty mushrooms | Canada, US, Netherlands, China |
| Ancient/heritage grains | Ethiopia, Peru, Canada |
| Adaptogens (ashwagandha, reishi) | India, China |

---

## Cluster D — Industrial Biomaterials & Fibre

Price co-moves with energy markets and policy signals (carbon credits, renewable fuel mandates). Margin is thin — logistics efficiency and contract structure determine profitability.

| Characteristic | Description |
|---|---|
| Price driver | Energy prices + policy/carbon incentives |
| Grade basis | Energy content (GJ/tonne), moisture, ash |
| Buyer type | Energy producers, manufacturers, carbon programs |
| Contract | Long-term offtake common |
| Model class | Energy proxy regression, policy regime switching |

| Market | Region |
|---|---|
| Wood chips & sawdust | BC, Scandinavia, Southeast Asia |
| Hemp fibre & shiv | Canada, EU, China |
| Pellet-grade biomass | US Southeast, Baltic states |
| Flax straw | Saskatchewan, Kazakhstan |
| Biochar feedstocks | Global — nascent |

---

## Cluster E — Circular Economy & By-Products

Price can be zero or negative — the seller may be paying to dispose. The arbitrage is finding a buyer in a different geography or application who will pay for the same material.

| Characteristic | Description |
|---|---|
| Price driver | Disposal cost vs alternative use value |
| Price range | Can be negative (disposal cost) to positive |
| Buyer type | Animal feed, bioenergy, compost, industrial processors |
| Liquidity | Extremely thin — relationship-driven |
| Model class | Disposal cost floor, spread to substitute commodity |

| Market | Region |
|---|---|
| Spent grain (brewery/distillery) | Global — near every brewery |
| Food processing waste (pomace, pulp) | US, EU, South America |
| Used cooking oil | Global |
| Digestate / biosolids | EU, Canada, Australia |
| Animal by-products (feathers, blood meal) | Global |

---

## Cluster F — Aquatic & Marine Outputs

Harvest seasonality similar to Cluster A, but with regulatory quota risk creating sudden supply discontinuities. Price can gap 30–50% on a licence decision.

| Characteristic | Description |
|---|---|
| Price driver | Harvest season + quota availability |
| Grade basis | Species, processing method, freshness |
| Risk | Regulatory quota — supply can stop abruptly |
| Buyer type | Feed manufacturers, food processors, fertiliser |
| Model class | OU + Poisson quota events, regime switching |

| Market | Region |
|---|---|
| Seaweed & kelp meal | Norway, Japan, Chile, Atlantic Canada |
| Fish meal & fish oil | Peru, Chile, Norway, Iceland |
| Shellfish meal | Atlantic Canada, New England, Japan |
| Duckweed protein | Nascent — Netherlands, Southeast Asia |
| Brine shrimp | Utah (Great Salt Lake), Kazakhstan |

---

## Cluster G — Functional Minerals & Earths

Geologically constrained supply with very few producing regions globally. Price is set by the dominant producer rather than by market clearing — index work here is intelligence, not price discovery.

| Characteristic | Description |
|---|---|
| Price driver | Producer pricing + logistics from source |
| Supply constraint | Geography — few producing regions exist |
| Grade basis | Purity, particle size, heavy metal content |
| Buyer type | Ag inputs, industrial, water treatment |
| Model class | Producer price proxy, transport basis, quality regression |

| Market | Region |
|---|---|
| Bentonite clay | Wyoming (US), Turkey, India |
| Zeolite | US, Slovakia, Japan |
| Diatomaceous earth | US, China, Mexico |
| Perlite | Turkey, Greece, US |
| Rock phosphate | Morocco, China, Russia |

---

## Cluster H — Animal Inputs & Feed Additives

Price is mechanically linked to the livestock cycle — cattle, hog, and poultry numbers drive demand. These markets move with the protein complex even when the physical product looks nothing like feed grain.

| Characteristic | Description |
|---|---|
| Price driver | Livestock numbers × feed conversion rates |
| Grade basis | Nutrient content (protein, energy, minerals) |
| Buyer type | Feed mills, integrators, livestock operations |
| Proxy | CME corn, soybean meal as liquid hedges |
| Model class | Proxy regression, livestock cycle factor |

| Market | Region |
|---|---|
| Dried distillers grains (DDGS) | US Corn Belt, Canada |
| Canola meal | Western Canada, EU |
| Sunflower meal | Ukraine, Russia, Argentina |
| Insect meal (BSFL) | EU, Canada — fast growing |
| Lemna / aquatic protein | Nascent global |

---

## Cluster I — Carbon & Environmental Credits (Physical Basis)

The financial credit market exists but the underlying physical commodity (the project input or output) is unpriced and illiquid. SIP's role is pricing the physical side, which anchors the credit value.

| Characteristic | Description |
|---|---|
| Price driver | Carbon credit price × project methodology |
| Grade basis | Verification standard (VCS, Gold Standard, ACR) |
| Buyer type | Corporates, compliance buyers, project developers |
| Linkage | Physical commodity underpins credit issuance |
| Model class | Credit proxy regression, project cost floor model |

| Market | Region |
|---|---|
| Biochar (physical tonnes) | Global — linked to carbon registries |
| Soil amendment inputs for carbon farming | North America, Australia, EU |
| Reforestation species (native seedlings) | Canada, Brazil, Southeast Asia |
| Wetland restoration inputs | North America, Southeast Asia |
| Rangeland carbon inputs | Alberta, Australia, Kenya |

---

## Cluster J — Water & Irrigation Inputs

Pricing is fragmented by jurisdiction — water rights, allocation trading, and physical delivery inputs all operate under different rules by region. Enormous total market, almost no price transparency.

| Characteristic | Description |
|---|---|
| Price driver | Allocation scarcity + jurisdiction rules |
| Grade basis | Water quality parameters (salinity, nutrients) |
| Buyer type | Irrigators, municipalities, industrial users |
| Risk | Regulatory — allocation rules change |
| Model class | Scarcity index, allocation spot price |

| Market | Region |
|---|---|
| Irrigation water allocation trading | Murray-Darling (AU), US West, Chile |
| Treated effluent for agriculture | Israel, US Southwest, Spain |
| Desalination output contracts | Middle East, coastal AU, Spain |
| Rainwater harvesting inputs | Sub-Saharan Africa, South Asia |
| Groundwater access rights | India, US Great Plains |

---

## Cluster K — Textile & Natural Fibre

Price driven by fashion and apparel industry procurement cycles layered on top of crop seasonality. Synthetics are the price ceiling — when oil is cheap, synthetics cap natural fibre prices.

| Characteristic | Description |
|---|---|
| Price driver | Fashion cycle + synthetic fibre substitute price |
| Grade basis | Staple length, micron, clean yield |
| Price ceiling | Synthetic fibre cost (oil-linked) |
| Buyer type | Textile mills, yarn spinners |
| Model class | Spread to synthetic proxy, grade premium surface |

| Market | Region |
|---|---|
| Raw wool (non-commodity grades) | Patagonia, Central Asia, NZ |
| Alpaca fibre | Peru, Bolivia, US boutique |
| Nettle fibre | Nepal, EU artisan |
| Ramie | China, Philippines |
| Bast fibres (jute, kenaf) | Bangladesh, India, West Africa |

---

## Cluster L — Emerging Protein Crops

New crop categories without established pricing infrastructure. First-mover index advantage is highest here — there is no reference price at all, so SIP can define the market.

| Characteristic | Description |
|---|---|
| Price driver | Protein demand growth + processing capacity constraint |
| Grade basis | Protein %, amino acid profile, GMO status |
| Market stage | Pre-commercial to early commercial |
| Buyer type | Food tech, alt-protein manufacturers |
| Model class | Cost-of-production floor, protein parity to soy |

| Market | Region |
|---|---|
| Faba beans | Alberta, UK, Australia, Ethiopia |
| Lentils (specialty colours) | Saskatchewan, Turkey, Nepal |
| Lupin | Western Australia, EU |
| Chickpeas (non-commodity grades) | Saskatchewan, India, Australia |
| Winged bean | Papua New Guinea, Southeast Asia |

---

## Cluster M — Fermentation & Microbial Inputs

Biological inputs whose value is measured in living organisms, not tonnes. Standard commodity pricing frameworks break down — you are pricing viability and efficacy, not weight.

| Characteristic | Description |
|---|---|
| Price driver | Efficacy (CFU/g, viability at delivery) |
| Grade basis | Strain specificity, live count, shelf life |
| Transport | Cold chain — logistics is the primary risk |
| Buyer type | Ag input companies, food manufacturers |
| Model class | Quality premium, cold chain cost model |

| Market | Region |
|---|---|
| Inoculants (rhizobium, mycorrhizae) | Canada, US, Brazil, Australia |
| Kefir grains | Eastern Europe, global boutique |
| Koji cultures | Japan, expanding globally |
| Sourdough starter cultures | EU, North America artisan |
| Biostimulant microbial inputs | Global ag |

---

## Cluster N — Exotic & Ceremonial Commodities

Extremely thin physical markets where cultural significance drives a price premium that has no fundamental anchor. Price is set by the highest-value buyer's willingness to pay, not by supply/demand clearing.

| Characteristic | Description |
|---|---|
| Price driver | Cultural significance + scarcity perception |
| Grade basis | Provenance, ritual certification, reputation of producer |
| Liquidity | Near zero — a few tonnes per year in some markets |
| Buyer type | High-end food, luxury, ceremonial, pharmaceutical |
| Model class | Auction price model, comparable transaction regression |

| Market | Region |
|---|---|
| Saffron (premium grades) | Iran, Spain, Kashmir |
| Ceremonial cacao | Guatemala, Ecuador, Mexico |
| Wild-harvest truffles | France, Italy, Balkans |
| Aged pu-erh tea | Yunnan, China |
| Manuka honey (UMF graded) | New Zealand |

---

## Cluster O — Waste Heat & Energy By-Products

The physical commodity is thermal or electrical energy that cannot be stored — it must be consumed at the point of generation or lost. Pricing requires real-time or near-real-time settlement.

| Characteristic | Description |
|---|---|
| Price driver | Spot energy price + transmission constraint |
| Storage | Zero — instantaneous settlement required |
| Buyer type | District heating operators, industrial users adjacent to source |
| Contract | Offtake with curtailment provisions |
| Model class | Spot energy proxy, transmission basis model |

| Market | Region |
|---|---|
| Industrial waste heat (data centres, smelters) | Nordics, Canada, Netherlands |
| Biogas from agricultural digesters | Germany, Denmark, Netherlands, Canada |
| Landfill gas | North America, EU |
| Geothermal direct heat | Iceland, New Zealand, Kenya |
| Tidal/run-of-river micro-hydro | BC, Norway, New Zealand |

---

## Cluster P — Soil Inputs & Agricultural Amendments

Price is driven by crop input economics — farmers buy when crop margins justify it, defer when they don't. Demand is therefore correlated with commodity grain prices, not just supply of the amendment itself.

| Characteristic | Description |
|---|---|
| Price driver | Crop margin economics + soil biology awareness |
| Grade basis | Nutrient analysis, biological activity, heavy metal limits |
| Buyer type | Farmers, co-ops, precision ag operators |
| Seasonality | Pre-planting demand spike (spring), fall application secondary |
| Model class | Crop margin proxy, seasonal demand model |

| Market | Region |
|---|---|
| Compost (certified organic) | North America, EU, Australia |
| Biosolids (treated municipal) | Canada, US, EU |
| Gypsum (agricultural) | US, Australia, Middle East |
| Glacial rock dust | Canada, Scandinavia |
| Worm castings (vermicompost) | Global niche |
| Seabird / bat guano | Peru, Pacific islands, Southeast Asia |
| Leonardite / humates | North Dakota, Leonardite (Spain) |

---

## Cluster Q — Animal Genetics & Biological Materials

Value is determined by genetic performance data, not physical weight or grade. Pricing requires a performance model — a bull's semen straw is priced off expected progeny differences (EPDs), not kg.

| Characteristic | Description |
|---|---|
| Price driver | Genetic performance metrics (EPDs, EBVs, yield data) |
| Grade basis | Breed registration, health certification, genetic testing |
| Storage | Cryogenic — indefinite shelf life changes supply dynamics |
| Buyer type | Livestock producers, breeding operations, genetic companies |
| Model class | Performance index model, genetic premium regression |

| Market | Region |
|---|---|
| Beef genetics (semen, embryos) | Canada, US, Australia, Brazil |
| Dairy genetics | Canada, Netherlands, New Zealand |
| Swine genetics | Canada, Denmark, US |
| Poultry parent stock | Global — highly consolidated |
| Sheep & goat genetics | Australia, NZ, UK |
| Aquaculture broodstock | Norway, Chile, Canada |
| Honeybee queens & packages | North America, Australia, EU |

---

## Cluster R — Pharmaceutical Botanicals & Plant Extracts

Regulatory status (approved vs unapproved ingredient) creates abrupt price discontinuities. The same plant material can be priced as a commodity feed ingredient or as a high-value pharmaceutical input depending on jurisdiction and end-use.

| Characteristic | Description |
|---|---|
| Price driver | Regulatory approval status + active compound content |
| Grade basis | Standardised extract potency (%, mg/g active) |
| Risk | Regulatory reclassification can collapse or create a market overnight |
| Buyer type | Pharma, nutraceutical, cosmetic manufacturers |
| Model class | Regulatory regime switch, active content quality premium |

| Market | Region |
|---|---|
| Cannabis (non-intoxicating CBD biomass) | Canada, US, EU, Colombia |
| Echinacea root | North America, Eastern Europe |
| Valerian root | Eastern Europe, China |
| Milk thistle seed | Eastern Europe, India, China |
| Boswellia resin | Ethiopia, India, Somalia |
| Kratom | Southeast Asia (regulatory risk highest here) |
| Elderberry | Eastern Europe, North America |

---

## Cluster S — Construction & Natural Building Materials

Geographically heavy and low value-to-weight — transport radius defines the market. Price is anchored to the cost of the conventional substitute (concrete, synthetic insulation), making these spread markets rather than outright price markets.

| Characteristic | Description |
|---|---|
| Price driver | Substitute material cost (concrete, synthetic insulation) |
| Grade basis | Structural spec, moisture content, certifications |
| Transport | Very short radius — 100–300km typical maximum |
| Buyer type | Contractors, developers, owner-builders |
| Model class | Substitute spread model, transport radius constraint |

| Market | Region |
|---|---|
| Hempcrete (hemp shiv + lime) | Canada, France, UK, Australia |
| Straw bale (construction grade) | North America, EU, Australia |
| Rammed earth / adobe inputs | US Southwest, Africa, Middle East |
| Bamboo (structural grade) | Southeast Asia, South America, East Africa |
| Cork | Portugal, Spain, North Africa |
| Thatching reed | Netherlands, UK, East Africa |
| Mycelium composites (inputs) | Global nascent |

---

## Cluster T — Non-Edible Horticultural Outputs

Prices are set by aesthetic and functional specifications that are hard to standardise. The same flower variety grown in different conditions commands a radically different price — provenance and growing method matter as much as the physical product.

| Characteristic | Description |
|---|---|
| Price driver | Aesthetic spec + seasonality + event-driven demand spikes |
| Grade basis | Stem length, bloom stage, variety, growing method |
| Demand spikes | Valentine's Day, Mother's Day, weddings — predictable but extreme |
| Perishability | Days to weeks — cold chain is essential |
| Model class | Event-spike model, grade premium, cold chain cost |

| Market | Region |
|---|---|
| Dried flowers & botanicals | Netherlands, France, India |
| Medicinal lavender | France, Bulgaria, Canada |
| Essential oil crops (rose, neroli, jasmine) | Bulgaria, Morocco, Egypt, India |
| Ornamental grass & foliage | Kenya, Netherlands, Colombia |
| Moss & lichen (horticultural) | Scandinavia, Pacific Northwest |
| Seed-bearing ornamentals | Netherlands, US |
| Peat alternatives (coir, wood fibre) | Sri Lanka, India, Nordics |

---

## Cluster U — Reclaimed & Recovered Industrial Materials

Supply is determined by industrial activity levels, not harvests or weather. Price is bounded above by virgin material cost and below by processing/handling cost. The market exists only when the spread between those two bounds is positive.

| Characteristic | Description |
|---|---|
| Price driver | Virgin material price minus processing cost |
| Supply driver | Industrial output levels — countercyclical to recession |
| Price bounds | Floor: handling cost. Ceiling: virgin material substitute |
| Buyer type | Reprocessors, manufacturers, construction |
| Model class | Spread to virgin material, industrial cycle proxy |

| Market | Region |
|---|---|
| Recovered cooking fat (tallow, lard) | Global — rendering industry |
| Post-consumer textile waste | EU, South Asia, China |
| Agricultural plastic film | EU, Canada, Australia — nascent infrastructure |
| Reclaimed wood (demolition timber) | North America, UK, Australia |
| Rubber crumb (end-of-life tires) | Global |
| Recovered glass cullet | EU, North America |
| Steel slag & fly ash | Global — industrial by-product |

---

## Cluster V — Traded Seeds & Propagation Material

The price includes embedded intellectual property — variety rights, plant breeders' rights, and certification requirements create legal premiums on top of physical commodity value. Uncertified seed is a different (lower) market entirely.

| Characteristic | Description |
|---|---|
| Price driver | Variety performance data + certification status |
| Grade basis | Germination rate, purity, variety certification, treated vs untreated |
| IP layer | Plant breeders' rights create legal price floors |
| Buyer type | Farmers, nurseries, seed distributors |
| Model class | Certified premium over uncertified, performance regression |

| Market | Region |
|---|---|
| Certified forage seed (non-commodity varieties) | Western Canada, NZ, EU |
| Vegetable seed (open-pollinated specialty) | US, Netherlands, Italy |
| Native prairie seed | Great Plains, Australia |
| Wildflower seed mixes | UK, Europe, North America |
| Turfgrass seed (specialty blends) | Oregon, Netherlands, NZ |
| Tree & shrub seed | Pacific Northwest, EU, China |
| Certified organic seed | Global niche |

---

## Cluster W — Traditional & Indigenous Food Systems

Markets operating within traditional or indigenous food economies where price is embedded in cultural reciprocity systems, not pure market clearing. External demand (premium retail, restaurants) creates a second price tier that SIP can help arbitrage.

| Characteristic | Description |
|---|---|
| Price driver | External premium demand vs internal subsistence value |
| Grade basis | Traditional preparation method, provenance, community certification |
| Buyer type | Premium retail, restaurants, cultural organisations |
| Complexity | Community consent and benefit-sharing are non-negotiable |
| Model class | Dual-tier price model, provenance premium |

| Market | Region |
|---|---|
| Wild rice (manoomin) | Great Lakes, Canada |
| Bison meat & by-products | North America |
| Bannock grain inputs | Canada |
| Kakadu plum | Northern Australia |
| Bush tucker inputs (wattleseed, quandong) | Australia |
| Moringa (smallholder) | Sub-Saharan Africa, South Asia |
| Quinoa (non-commodity heritage varieties) | Bolivia, Peru, Andean communities |

---

## Cluster X — Logistics & Capacity Contracts

Not a commodity in the physical sense — what is being traded is access to infrastructure (storage, transport, processing capacity) at a future date. Pricing is purely forward-looking, driven by capacity utilisation expectations.

| Characteristic | Description |
|---|---|
| Price driver | Capacity utilisation forecasts + seasonal demand peaks |
| Settlement | Cash or physical delivery of service |
| Risk | Counterparty risk — seller may not have capacity when needed |
| Buyer type | Commodity traders, shippers, processors |
| Model class | Capacity option pricing, utilisation forward curve |

| Market | Region |
|---|---|
| Grain elevator storage contracts | Canadian prairies, US Midwest |
| Truck & rail capacity forwards | North America, Australia |
| Cold storage capacity | Global — protein and produce logistics |
| Port terminal capacity | Global bulk commodity ports |
| Custom crushing / processing slots | Wine regions, oil seed crush |
| Drying capacity (grain, biomass) | North America, EU |

---

## Cluster Y — Prediction & Outcome Markets (Physical Delivery Basis)

Markets where the traded instrument is a forecast or outcome, but where physical delivery of an input or output occurs contingent on that outcome. Weather derivatives are the closest liquid analogue — SIP's version is the physical commodity equivalent.

| Characteristic | Description |
|---|---|
| Price driver | Probability-weighted outcome × physical delivery value |
| Settlement | Physical delivery if trigger condition met |
| Counterparty | Insurers, reinsurers, ag lenders, crop finance |
| Complexity | Requires index + trigger definition + physical contract |
| Model class | Binary option, parametric insurance pricing |

| Market | Region |
|---|---|
| Parametric crop insurance inputs | Global ag — nascent physical side |
| Yield-contingent supply contracts | North America, Australia |
| Drought-triggered feed delivery contracts | Alberta, Australia, East Africa |
| Flood-contingent infrastructure inputs | South Asia, Southeast Asia |
| Frost-triggered fruit crop contracts | Canada, Chile, New Zealand |

---

## Cluster Z — Artisanal & Small-Batch Processing Outputs

Volume too small for commodity markets, quality too high for commodity pricing, and provenance too specific for generic grade systems. Price discovery is fragmented across direct relationships, artisan markets, and online platforms — SIP aggregates and publishes.

| Characteristic | Description |
|---|---|
| Price driver | Craft premium + provenance + producer reputation |
| Volume | Too small for commodity infrastructure — tonnes to kg |
| Grade basis | Artisan certification, method (cold-pressed, stone-ground, etc.) |
| Buyer type | Specialty retailers, restaurants, direct consumers |
| Model class | Comparable transaction regression, artisan premium index |

| Market | Region |
|---|---|
| Artisan cheese inputs (milk, cultures, rennet) | EU, North America, Australia |
| Cold-pressed oils (specialty crops) | Global |
| Small-batch vinegar (non-wine) | EU, Japan, North America |
| Artisan salt (fleur de sel, smoked) | France, Portugal, Japan |
| Raw honey (varietal, single-source) | Global |
| Traditional fermented inputs (miso, tempeh) | Japan, Indonesia, global expanding |
| Hand-harvested sea vegetables | Atlantic Canada, Japan, Ireland |

---

## Cluster Summary

| Cluster | Mechanism | SIP priority |
|---|---|---|
| A — Seasonal ag bulk | OU + harvest seasonality | Now — core business |
| B — Weather-event demand | Poisson jumps + regime switch | Near-term — municipal counterparties |
| C — Specialty food & beverage | Quality premium + origin basis | Near-term — high margin |
| D — Industrial biomaterials | Energy proxy + policy regime | Medium-term |
| E — Circular & by-products | Disposal cost floor + spread | Near-term — low capital, high arb |
| F — Aquatic & marine | OU + quota event risk | Medium-term |
| G — Functional minerals | Producer price proxy | Long-term |
| H — Animal inputs & feed additives | Livestock cycle proxy | Medium-term |
| I — Carbon & environmental | Credit proxy + project cost floor | Medium-term — policy risk |
| J — Water & irrigation | Scarcity index + jurisdiction | Long-term — regulatory complexity |
| K — Textile & natural fibre | Synthetic spread + grade surface | Medium-term |
| L — Emerging protein crops | Cost-of-production + protein parity | Near-term — first-mover index advantage |
| M — Fermentation & microbial | Efficacy premium + cold chain | Long-term |
| N — Exotic & ceremonial | Auction + comparable transaction | Opportunistic |
| O — Waste heat & energy by-products | Spot energy proxy + transmission basis | Long-term |
| P — Soil inputs & amendments | Crop margin proxy + seasonal demand | Near-term — natural adjacency to Cluster A |
| Q — Animal genetics & biologics | Genetic performance index | Medium-term |
| R — Pharmaceutical botanicals | Regulatory regime switch + potency premium | Medium-term — regulatory risk |
| S — Construction & natural building | Substitute spread + transport radius | Long-term |
| T — Non-edible horticultural | Event spike + grade premium | Medium-term |
| U — Reclaimed & recovered industrial | Spread to virgin material + industrial cycle | Medium-term |
| V — Traded seeds & propagation | Certified premium + IP layer | Medium-term |
| W — Traditional & indigenous food | Dual-tier price + provenance premium | Long-term — community complexity |
| X — Logistics & capacity contracts | Capacity option + utilisation forward | Medium-term — infrastructure adjacency |
| Y — Prediction & outcome markets | Parametric/binary option pricing | Long-term — structural complexity |
| Z — Artisanal & small-batch | Comparable transaction regression | Opportunistic |

---

*SIP Global — working document v2.0 — 26 clusters*
