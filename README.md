app_lifegame

Adatpárhuzamos programozás - GPU

Legyen GPU-n gyorsított, az indított szálak száma legyen több, mint 1000-es nagyságrendű (több block)

Mátrixban minden cellában három jelölés valamelyike lehet: élő sejt, üres hely és gyógyszer. Ha egy üres cella szomszédjában pontosan 3 élő sejt van, akkor a következő generációba erre a helyre egy élő sejt kerül. Ha egy élő sejt mellett kevesebb mint 2 vagy több mint 3 sejt van akkor az adott sejt "meghal", a helyén üres cella lesz. Egy cellának 8 szomszédja van. Emellett néhány cellában lehet gyógyszer is. A gyógyszer a közvetlenül szomszédos sejteket megöli kivéve, ha pontosan 3 sejt van a szomszédjában.
A feladathoz a kezdőmátrixot a CPU oldalon készülne. Továbbá egy számolásnál több generáción is túljutunk, ezt is CPU oldalon kell megadni. A mátrix változtatások történnek GPU-n, majd a mátrixot akkor íródik vissza CPU-ra, amikor annyiszor számoltuk újra, ahányat CPU-n generáció számnak megadtunk.
