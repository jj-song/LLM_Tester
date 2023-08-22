using System;
using System.Data;

public class UserService
{
    private readonly IUserRepository _userRepository;
    private readonly ILogger _logger;

    public UserService(IUserRepository userRepository, ILogger logger)
    {
        _userRepository = userRepository;
        _logger = logger;
    }

    public User GetUserById(int id)
    {
        try
        {
            _logger.LogInfo($"Fetching user with ID: {id}");
            User user = _userRepository.GetById(id);

            if (user == null)
            {
                _logger.LogWarning($"User with ID: {id} not found");
                return null;
            }

            // Apply some business logic, e.g., calculate age
            user.Age = CalculateAge(user.DateOfBirth);

            return user;
        }
        catch (DataException ex)
        {
            _logger.LogError($"Database error while fetching user with ID: {id}", ex);
            throw new Exception("An error occurred while fetching the user", ex);
        }
        catch (Exception ex)
        {
            _logger.LogError($"Unexpected error while fetching user with ID: {id}", ex);
            throw;
        }
    }

    private int CalculateAge(DateTime dateOfBirth)
    {
        int age = DateTime.Today.Year - dateOfBirth.Year;
        if (dateOfBirth.Date > DateTime.Today.AddYears(-age)) age--;
        return age;
    }
}

public interface IUserRepository
{
    User GetById(int id);
}

public interface ILogger
{
    void LogInfo(string message);
    void LogWarning(string message);
    void LogError(string message, Exception ex);
}

public class User
{
    public int Id { get; set; }
    public string Name { get; set; }
    public DateTime DateOfBirth { get; set; }
    public int Age { get; set; }
}
