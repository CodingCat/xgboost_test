package me.codingcat.rosseman

//"Store","DayOfWeek","Date","Sales","Customers","Open","Promo","StateHoliday","SchoolHoliday"
case class SalesRecord(storeId: Int, daysOfWeek: Int, date: String, sales: Int, customers: Int,
                       open: Int, promo: Int, stateHoliday: String, schoolHoliday: String)
//"Store","StoreType","Assortment","CompetitionDistance","CompetitionOpenSinceMonth",
// "CompetitionOpenSinceYear","Promo2","Promo2SinceWeek","Promo2SinceYear","PromoInterval"
case class Store(storeId: Int, storeType: String, assortment: String, competitionDistance: Int,
                 competitionOpenSinceMonth: Int, competitionOpenSinceYear: Int, promo2: Int,
                 promo2SinceWeek: Int, promo2SinceYear: Int, promoInterval: String)
