using SpacetimeDB;
using System.Linq;

public static partial class Module
{
    [SpacetimeDB.Table(Accessor = "Room", Public = true)]
    public partial struct Room
    {
        [SpacetimeDB.PrimaryKey]
        public string Code;
        public Identity P1;
        public Identity? P2;
        public uint P1Progress;
        public uint P2Progress;
    }

    [SpacetimeDB.Reducer]
    public static void CreateRoom(ReducerContext ctx)
    {
        const string chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
        string code;
        do
        {
            code = new string(Enumerable.Range(0, 5)
                .Select(_ => chars[ctx.Rng.Next(chars.Length)]).ToArray());
        } while (ctx.Db.Room.Code.Find(code) is not null);

        ctx.Db.Room.Insert(new Room
        {
            Code = code,
            P1 = ctx.Sender,
            P2 = null,
            P1Progress = 0,
            P2Progress = 0
        });
    }

    [SpacetimeDB.Reducer]
    public static void JoinRoom(ReducerContext ctx, string code)
    {
        var found = ctx.Db.Room.Code.Find(code);
        if (found is null) throw new Exception("No such room");

        var room = found.Value;
        if (room.P2 is not null) throw new Exception("Room is full");

        room.P2 = ctx.Sender;
        ctx.Db.Room.Code.Update(room);
    }

    [SpacetimeDB.Reducer]
    public static void SetProgress(ReducerContext ctx, string code, uint index)
    {
        var found = ctx.Db.Room.Code.Find(code);
        if (found is null) throw new Exception("No such room");

        var room = found.Value;
        if (ctx.Sender == room.P1) room.P1Progress = index;
        else if (room.P2 is not null && ctx.Sender == room.P2.Value) room.P2Progress = index;
        else throw new Exception("Not in this room");

        ctx.Db.Room.Code.Update(room);
    }
}