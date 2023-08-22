public class ShoppingCart
{
    public List<Item> Items { get; set; }

    public ShoppingCart()
    {
        Items = new List<Item>();
    }

    public double CalculateTotalPrice(bool applyDiscount)
    {
        double total = 0;

        foreach (var item in Items)
        {
            total += item.Price * item.Quantity;
        }

        if (applyDiscount)
        {
            total -= ApplyDiscount(total);
        }

        return total;
    }

    private double ApplyDiscount(double total)
    {
        // Apply a 10% discount if the total price is greater than 100
        return total > 100 ? total * 0.10 : 0;
    }
}

public class Item
{
    public string Name { get; set; }
    public double Price { get; set; }
    public int Quantity { get; set; }
}
